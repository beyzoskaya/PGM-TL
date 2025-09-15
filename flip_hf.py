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
    """Base class for loading HuggingFace protein datasets"""
    
    def load_hf_dataset(self, dataset_name, split_column='split', sequence_column='sequence', 
                       target_columns=None, verbose=1):
        """Load dataset from HuggingFace"""
        if verbose:
            print(f"Loading dataset: {dataset_name}")
        
        # Load the dataset
        dataset = load_dataset(dataset_name)
        
        # Handle different split formats
        if 'train' in dataset:
            # Dataset already has splits
            train_data = dataset['train']
            valid_data = dataset.get('validation', dataset.get('valid', []))
            test_data = dataset.get('test', [])
        else:
            # Single dataset, need to split based on column
            all_data = dataset['train'] if 'train' in dataset else dataset
            train_data = [item for item in all_data if item.get(split_column) == 'train']
            valid_data = [item for item in all_data if item.get(split_column) in ['valid', 'validation']]
            test_data = [item for item in all_data if item.get(split_column) == 'test']
        
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
                                # Token-level targets (e.g., secondary structure)
                                all_targets[col].append(value)  # Keep as list
                            else:
                                # Sequence-level targets (e.g., stability)
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


class AAV(HuggingFaceDataset):
    """AAV dataset using HuggingFace datasets"""
    
    target_fields = ["target"]
    region = slice(474, 674)
    
    def __init__(self, path, split="two_vs_many", keep_mutation_region=False, verbose=1, **kwargs):
        super().__init__()
        
        # For AAV, we can try to find a suitable dataset or create our own mapping
        # This is a placeholder - you might need to upload the FLIP AAV data to HF or use a different approach
        try:
            dataset_name = "proteinglm/aav_prediction"  # This might not exist, just an example
            self.load_hf_dataset(dataset_name, target_columns=self.target_fields, verbose=verbose)
        except:
            # Fallback: you could manually process the FLIP data and convert to HF format
            print(f"Could not load AAV dataset from HuggingFace. You may need to:")
            print(f"1. Upload the FLIP AAV data to HuggingFace Hub")
            print(f"2. Or manually convert the CSV files to HuggingFace format")
            print(f"3. Or use datasets.Dataset.from_csv() to load local CSV files")
            raise ValueError("AAV dataset not available in HuggingFace format")
        
        if keep_mutation_region:
            # Apply region slicing to sequences
            self.sequences = [seq[self.region] if len(seq) > self.region.stop else seq 
                            for seq in self.sequences]
    
    def split(self):
        """Split dataset into train/valid/test"""
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            if num_sample > 0:
                split = Subset(self, range(offset, offset + num_sample))
                splits.append(split)
                offset += num_sample
            else:
                splits.append(Subset(self, []))  # Empty subset
        return splits


class GB1(HuggingFaceDataset):
    """GB1 dataset using HuggingFace datasets"""
    
    target_fields = ["target"]
    
    def __init__(self, path, split="two_vs_rest", verbose=1, **kwargs):
        super().__init__()
        
        # Try to load from HuggingFace
        try:
            dataset_name = "proteinglm/gb1_prediction"  # This might not exist
            self.load_hf_dataset(dataset_name, target_columns=self.target_fields, verbose=verbose)
        except:
            print("GB1 dataset not found in HuggingFace format")
            raise ValueError("GB1 dataset not available in HuggingFace format")
    
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


# Alternative: Direct HuggingFace dataset loader
class SecondaryStructure(HuggingFaceDataset):
    """Secondary Structure Prediction using proteinglm/ssp_q8"""
    
    target_fields = ["ss8"]
    
    def __init__(self, path, verbose=1, **kwargs):
        super().__init__()
        
        try:
            dataset_name = "proteinglm/ssp_q8"
            self.load_hf_dataset(dataset_name, 
                               sequence_column='seq',
                               target_columns=['label'],
                               verbose=verbose)
            
            # Convert secondary structure strings to lists if needed
            if 'ss8' in self.targets:
                self.targets['target'] = self.targets['ss8']
                
        except Exception as e:
            print(f"Error loading secondary structure dataset: {e}")
            raise ValueError("Secondary structure dataset not available")
    
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
        'AAV': AAV,
        'GB1': GB1,
        'Thermostability': Thermostability,
        'SecondaryStructure': SecondaryStructure,
        'PeptideHLAMHCAffinity': PeptideHLAMHCAffinity
    }
    
    if dataset_type not in datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return datasets[dataset_type](**kwargs)