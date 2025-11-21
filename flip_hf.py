import torch
from torch.utils.data import Dataset, Subset
from datasets import load_dataset
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class ProteinDataset(Dataset):
    def __init__(self):
        self.sequences = []
        self.targets = {}
        self.num_samples = [0, 0, 0]
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target_dict = {}
        for key, values in self.targets.items():
            target_dict[key] = values[idx]
        return {'sequence': seq, 'targets': target_dict}

class HuggingFaceDataset(ProteinDataset):
    def load_hf_dataset(self, dataset_name, sequence_column='sequence', target_columns=None, verbose=1):
        if target_columns is None: target_columns = []
        dataset = load_dataset(dataset_name)
        
        # Basic split logic (simplified for brevity)
        if 'train' in dataset:
            train = dataset['train']
            valid = dataset.get('validation', dataset.get('valid'))
            test = dataset.get('test')
            
            if not valid and not test:
                split = train.train_test_split(test_size=0.2, seed=42)
                train = split['train']
                temp = split['test']
                split2 = temp.train_test_split(test_size=0.5, seed=42)
                valid, test = split2['train'], split2['test']
            elif not valid:
                split = train.train_test_split(test_size=0.1, seed=42)
                train, valid = split['train'], split['test']
                
            splits = [train, valid, test]
        else:
            # Fallback if structure is weird
            all_data = dataset[list(dataset.keys())[0]]
            split = all_data.train_test_split(test_size=0.2, seed=42)
            train = split['train']
            temp = split['test']
            split2 = temp.train_test_split(test_size=0.5, seed=42)
            splits = [train, split2['train'], split2['test']]

        self.num_samples = [len(s) for s in splits]
        self.sequences = []
        self.targets = {col: [] for col in target_columns}

        for split in splits:
            for item in split:
                seq = item.get(sequence_column) or item.get('seq') or ""
                self.sequences.append(seq)
                for col in target_columns:
                    self.targets[col].append(item.get(col, None))
        
        if verbose:
            print(f"✓ Loaded {dataset_name}: Train={self.num_samples[0]}, Val={self.num_samples[1]}, Test={self.num_samples[2]}")

    def split(self):
        offset = 0
        splits = []
        for n in self.num_samples:
            splits.append(Subset(self, range(offset, offset + n)))
            offset += n
        return splits

class Thermostability(HuggingFaceDataset):
    """Regression: Raw Targets (Normalization handled in Main)"""
    target_fields = ["label"]

    def __init__(self, verbose=1, **kwargs):
        super().__init__()
        self.load_hf_dataset("proteinglm/stability_prediction", sequence_column='seq', target_columns=self.target_fields, verbose=verbose)
        
        self.targets['target'] = [float(x) if x is not None else None for x in self.targets.pop('label')]
        
        if verbose:
            print(f"  [Info] Loaded raw thermostability targets (will normalize in main).")

class SecondaryStructure(HuggingFaceDataset):
    """Token Classification"""
    target_fields = ["label"]

    def __init__(self, verbose=1, **kwargs):
        super().__init__()
        self.load_hf_dataset("proteinglm/ssp_q8", sequence_column='seq', target_columns=self.target_fields, verbose=verbose)
        
        processed = []
        raw = self.targets.pop('label')
        
        for idx, item in enumerate(raw):
            if item is None:
                processed.append([])
                continue
            
            # Handle string or list input
            vals = []
            source = list(item) if isinstance(item, str) else item
            for x in source:
                try:
                    v = int(x)
                    vals.append(v if 0 <= v <= 7 else -100) # -100 ignored index
                except:
                    vals.append(-100)
            
            # Truncate/Pad logic handled in collate, but we check length here
            if len(vals) != len(self.sequences[idx]):
                # Mismatch fallback: trunctate to shorter
                min_len = min(len(vals), len(self.sequences[idx]))
                vals = vals[:min_len]
            
            processed.append(vals)
            
        self.targets['target'] = processed

class CloningCLF(HuggingFaceDataset):
    """Binary Classification"""
    target_fields = ["label"]

    def __init__(self, verbose=1, **kwargs):
        super().__init__()
        self.load_hf_dataset("proteinglm/cloning_clf", sequence_column='seq', target_columns=self.target_fields, verbose=verbose)
        self.targets['target'] = [int(x) if x is not None else None for x in self.targets.pop('label')]

def create_dataset(name, **kwargs):
    if name == 'Thermostability': return Thermostability(**kwargs)
    if name == 'SecondaryStructure': return SecondaryStructure(**kwargs)
    if name == 'CloningCLF': return CloningCLF(**kwargs)
    raise ValueError(f"Unknown dataset {name}")