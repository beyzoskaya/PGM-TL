import os
import math
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Subset
from datasets import load_dataset
from transformers import BertTokenizer


class ProteinDataset(Dataset):
    """Base protein dataset class to replace torchdrug.data.ProteinDataset"""
    
    def __init__(self):
        self.sequences = []
        self.targets = {}
        self.data = []
        self.num_samples = [0, 0, 0]  # train, valid, test
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target_dict = {}
        for key, values in self.targets.items():
            target_dict[key] = values[idx] if idx < len(values) else 0.0
            
        return {
            'sequence': sequence,
            'targets': target_dict,
            'graph': self.create_protein_graph(sequence)
        }
    
    def create_protein_graph(self, sequence):
        """Create a simple protein representation"""
        # Map amino acids to integers
        aa_to_id = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
            'X': 20, 'U': 21, 'B': 22, 'Z': 23, 'O': 24  # Extended amino acids
        }
        
        # Convert sequence to residue types
        residue_type = []
        for aa in sequence:
            residue_type.append(aa_to_id.get(aa, 20))  # Default to X for unknown
        
        return {
            'residue_type': torch.tensor(residue_type, dtype=torch.long),
            'num_residues': torch.tensor([len(residue_type)], dtype=torch.long),
            'batch_size': 1
        }


class HuggingFaceDataset(ProteinDataset):

    def load_hf_dataset(self, dataset_name, split_column='split', sequence_column='sequence', 
                       target_columns=None, verbose=1, valid_ratio=0.1):
        if verbose:
            print(f"Loading dataset: {dataset_name}")
            for key, values in self.targets.items():
                print(f"DEBUG: First 2 target values for {key}:", values[:2])
        
        # Load the dataset
        dataset = load_dataset(dataset_name)
        
        # Handle splits
        if 'train' in dataset:
            train_data = dataset['train']
            valid_data = dataset.get('validation', dataset.get('valid', []))
            test_data = dataset.get('test', [])
            
            # If validation is missing or empty, create a split from train
            if len(valid_data) == 0:
                if verbose:
                    print(f"No validation split found. Splitting {len(train_data)} training samples...")
                split_data = train_data.train_test_split(test_size=valid_ratio, seed=42)
                train_data = split_data['train']
                valid_data = split_data['test']
        else:
            # Single dataset, split based on column
            all_data = dataset['train'] if 'train' in dataset else dataset
            train_data = [item for item in all_data if item.get(split_column) == 'train']
            valid_data = [item for item in all_data if item.get(split_column) in ['valid', 'validation']]
            test_data = [item for item in all_data if item.get(split_column) == 'test']
            
            # If validation is missing, split from train
            if len(valid_data) == 0:
                if verbose:
                    print(f"No validation split found. Splitting {len(train_data)} training samples...")
                n_valid = max(1, int(len(train_data) * valid_ratio))
                valid_data = train_data[:n_valid]
                train_data = train_data[n_valid:]
        
        # Combine all data in order: train, valid, test
        all_sequences = []
        all_targets = defaultdict(list)
        
        for data_split in [train_data, valid_data, test_data]:
            for item in data_split:
                all_sequences.append(item[sequence_column])
                
                if target_columns:
                    for col in target_columns:
                        if col in item:
                            value = item[col]
                            if value == "" or value is None:
                                value = float('nan')
                            # Handle different target types properly
                            if isinstance(value, list):
                                all_targets[col].append(value)
                            else:
                                try:
                                    all_targets[col].append(float(value))
                                except (ValueError, TypeError):
                                    all_targets[col].append(float('nan'))
                        else:
                            all_targets[col].append(float('nan'))
        
        self.sequences = all_sequences
        self.targets = dict(all_targets)
        self.num_samples = [len(train_data), len(valid_data), len(test_data)]
        
        if verbose:
            print(f"Loaded {len(all_sequences)} sequences")
            print(f"Splits - Train: {self.num_samples[0]}, Valid: {self.num_samples[1]}, Test: {self.num_samples[2]}")

class Thermostability(HuggingFaceDataset):
    """Thermostability dataset using HuggingFace datasets"""
    
    target_fields = ["target"]
    
    def __init__(self, path, split="human_cell", verbose=1, **kwargs):
        super().__init__()
        
        # Use the proteinglm stability prediction dataset
        try:
            dataset_name = "proteinglm/stability_prediction"
            self.load_hf_dataset(dataset_name, 
                               sequence_column='seq',
                               target_columns=['label'],  
                               verbose=verbose)
            
            # Map stability_score to target for compatibility
            if 'stability_score' in self.targets:
                self.targets['target'] = self.targets['stability_score']
                
        except Exception as e:
            print(f"Error loading thermostability dataset: {e}")
            raise ValueError("Thermostability dataset not available in HuggingFace format")
    
    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            if num_sample > 0:
                split = Subset(self, range(offset, offset + num_sample))
                splits.append(split)
                offset += num_sample
            else:
                splits.append(Subset(self, []))
        return splits


class SecondaryStructure(HuggingFaceDataset):
    """Secondary Structure Prediction using proteinglm/ssp_q8"""
    
    target_fields = ["label"] 
    
    def __init__(self, path=None, verbose=1, **kwargs):
        super().__init__()
        
        try:
            dataset_name = "proteinglm/ssp_q8"
            
            # Load dataset with auto-validation split
            self.load_hf_dataset(
                dataset_name,
                sequence_column='seq',
                target_columns=self.target_fields,
                verbose=verbose
            )
            
            # Ensure the 'target' key exists and contains the label data
            if self.target_fields[0] in self.targets:
                # Convert to list of characters if needed
                processed_targets = []
                for item in self.targets[self.target_fields[0]]:
                    if isinstance(item, str):
                        processed_targets.append(list(item))
                    else:
                        processed_targets.append(item)
                self.targets['target'] = processed_targets
                print("DEBUG: First 2 token labels:", self.targets[self.target_fields[0]][:2])
            else:
                print("Warning: 'label' field not found in dataset. Setting empty targets.")
                self.targets['target'] = [[] for _ in range(len(self.sequences))]
        
        except Exception as e:
            print(f"Error loading secondary structure dataset: {e}")
            raise ValueError("Secondary structure dataset not available")
    
    def split(self):
        """Split dataset into train/valid/test subsets"""
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            if num_sample > 0:
                split = Subset(self, range(offset, offset + num_sample))
                splits.append(split)
                offset += num_sample
            else:
                splits.append(Subset(self, []))
        return splits

class PeptideHLAMHCAffinity(HuggingFaceDataset):
    """Peptide HLA MHC Affinity prediction using proteinglm/peptide_HLA_MHC_affinity"""
    
    target_fields = ["label"]
    
    def __init__(self, path, verbose=1, **kwargs):
        super().__init__()
        
        try:
            dataset_name = "proteinglm/peptide_HLA_MHC_affinity"
            self.load_hf_dataset(dataset_name, 
                               sequence_column='seq',
                               target_columns=['label'],
                               verbose=verbose)
            
            # Map label to target for compatibility
            if 'label' in self.targets:
                self.targets['target'] = self.targets['label']
                
        except Exception as e:
            print(f"Error loading proteinglm/peptide_HLA_MHC_affinity dataset: {e}")
            print("Please check if the dataset exists and column names are correct")
            raise ValueError("Peptide HLA MHC affinity dataset not available")
    
    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            if num_sample > 0:
                split = Subset(self, range(offset, offset + num_sample))
                splits.append(split)
                offset += num_sample
            else:
                splits.append(Subset(self, []))
        return splits

def create_dataset(dataset_type, **kwargs):
    """Factory function to create datasets"""
    datasets = {
        'Thermostability': Thermostability,
        'SecondaryStructure': SecondaryStructure,
        'PeptideHLAMHCAffinity': PeptideHLAMHCAffinity
    }
    
    if dataset_type not in datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return datasets[dataset_type](**kwargs)