import os
import math
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Subset
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

class ProteinDataset(Dataset):
    """
    Raw sequences and targets - no tokenization
    (tokenization is done in collate_fn)
    """

    def __init__(self):
        self.sequences = []
        self.targets = {}
        self.num_samples = [0, 0, 0]  # train/val/test split sizes
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target_dict = {}
        for key, values in self.targets.items():
            if idx < len(values):
                target_dict[key] = values[idx]
            else:
                target_dict[key] = None
        
        return {
            'sequence': seq,
            'targets': target_dict,
        }

    def dataset_summary(self):
        print(f"Sequences: {len(self.sequences)}; Targets: {list(self.targets.keys())}")


class HuggingFaceDataset(ProteinDataset):
    
    def load_hf_dataset(self, dataset_name, sequence_column='sequence', target_columns=None,
                        verbose=1, valid_ratio=0.1, test_ratio=0.1):
        """Load HuggingFace dataset and split into train/valid/test"""
        if target_columns is None:
            target_columns = []

        dataset = load_dataset(dataset_name)

        # Extract splits
        if isinstance(dataset, dict) and 'train' in dataset:
            train_data = dataset['train']
            valid_data = dataset.get('validation', dataset.get('valid', None))
            test_data = dataset.get('test', None)

            if valid_data is not None and test_data is not None:
                if verbose:
                    print(f"✓ Using provided TRAIN / VALID / TEST splits.")

            elif valid_data is None and test_data is not None:
                if verbose:
                    print(f"⚠ TRAIN + TEST found — splitting VALIDATION from TRAIN")
                split = train_data.train_test_split(test_size=valid_ratio, seed=42)
                train_data = split['train']
                valid_data = split['test']

            else:
                if verbose:
                    print(f"⚠ Only TRAIN found — creating VALID & TEST splits")
                split1 = train_data.train_test_split(test_size=(valid_ratio + test_ratio), seed=42)
                train_data = split1['train']
                temp = split1['test']
                relative_test_ratio = test_ratio / (valid_ratio + test_ratio)
                split2 = temp.train_test_split(test_size=relative_test_ratio, seed=42)
                valid_data = split2['train']
                test_data = split2['test']
        else:
            all_data = list(dataset.values())[0] if isinstance(dataset, dict) else dataset
            split1 = all_data.train_test_split(test_size=(valid_ratio + test_ratio), seed=42)
            train_data = split1['train']
            temp = split1['test']
            relative_test_ratio = test_ratio / (valid_ratio + test_ratio)
            split2 = temp.train_test_split(test_size=relative_test_ratio, seed=42)
            valid_data = split2['train']
            test_data = split2['test']

        splits = [train_data, valid_data, test_data]
        all_sequences = []
        all_targets = defaultdict(list)

        for split in splits:
            for item in split:
                seq = item.get(sequence_column) or item.get('seq') or ""
                all_sequences.append(seq)
                for col in target_columns:
                    val = item.get(col, None)
                    all_targets[col].append(val)

        self.sequences = all_sequences
        for col in target_columns:
            self.targets[col] = all_targets[col]
        self.num_samples = [len(train_data), len(valid_data), len(test_data)]

        if verbose:
            print(f"✓ Loaded dataset '{dataset_name}'")
            print(f"  TRAIN: {self.num_samples[0]}")
            print(f"  VALID: {self.num_samples[1]}")
            print(f"  TEST:  {self.num_samples[2]}")
            print(f"  Targets: {target_columns}")

    def split(self):
        """Return train/valid/test subsets"""
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            if num_sample > 0:
                splits.append(Subset(self, range(offset, offset + num_sample)))
                offset += num_sample
            else:
                splits.append(Subset(self, []))
        return splits


class Thermostability(HuggingFaceDataset):
    """Regression dataset - targets are floats"""
    target_fields = ["label"]

    def __init__(self, verbose=1, **kwargs):
        super().__init__()
        try:
            self.load_hf_dataset("proteinglm/stability_prediction", 
                                sequence_column='seq', 
                                target_columns=self.target_fields, 
                                verbose=verbose)
            
            # Normalize to 'target' key
            if 'label' in self.targets:
                self.targets['target'] = [None if v is None else float(v) for v in self.targets.pop('label')]
            else:
                self.targets['target'] = [None] * len(self.sequences)

            # Compute train mean/std
            train_vals = [v for v in self.targets['target'][:self.num_samples[0]] if v is not None]
            if len(train_vals) > 0:
                self.reg_mean = float(np.mean(train_vals))
                self.reg_std = float(np.std(train_vals)) if float(np.std(train_vals)) > 0 else 1.0
            else:
                self.reg_mean = 0.0
                self.reg_std = 1.0

            if verbose:
                print(f"  regression mean/std (train): {self.reg_mean:.4f}/{self.reg_std:.4f}")

        except Exception as e:
            logger.exception("Error loading thermostability dataset")
            raise


class SecondaryStructure(HuggingFaceDataset):
    """Token-level classification dataset - targets are lists of ints"""
    target_fields = ["label"]

    def __init__(self, verbose=1, **kwargs):
        super().__init__()
        try:
            self.load_hf_dataset("proteinglm/ssp_q8", 
                                sequence_column='seq', 
                                target_columns=self.target_fields, 
                                verbose=verbose)

            processed_targets = []
            invalid_count = 0
            length_mismatch_count = 0

            raw_labels = self.targets.get('label', [None] * len(self.sequences))
            for idx, item in enumerate(raw_labels):
                if item is None:
                    processed_targets.append([])
                    continue

                if isinstance(item, str):
                    chars = list(item.strip())
                    vals = []
                    for ch in chars:
                        try:
                            v = int(ch)
                            if v < 0 or v > 7:
                                invalid_count += 1
                                v = 0
                        except Exception:
                            invalid_count += 1
                            v = 0
                        vals.append(v)
                    seq_len = len(self.sequences[idx]) if idx < len(self.sequences) else 0
                    if len(vals) != seq_len:
                        length_mismatch_count += 1
                    processed_targets.append(vals)

                elif isinstance(item, (list, tuple)):
                    vals = []
                    for v in item:
                        try:
                            iv = int(v)
                            if iv < 0 or iv > 7:
                                invalid_count += 1
                                iv = 0
                        except Exception:
                            invalid_count += 1
                            iv = 0
                        vals.append(iv)
                    processed_targets.append(vals)
                else:
                    processed_targets.append([])

            self.targets['target'] = processed_targets
            if 'label' in self.targets:
                self.targets.pop('label', None)

            if verbose:
                print(f"  Invalid labels fixed: {invalid_count}")
                print(f"  Label/sequence length mismatches: {length_mismatch_count}")

        except Exception as e:
            logger.exception("Error loading secondary structure dataset")
            raise


class CloningCLF(HuggingFaceDataset):
    """Binary classification dataset - targets are ints"""
    target_fields = ["label"]

    def __init__(self, verbose=1, **kwargs):
        super().__init__()
        try:
            self.load_hf_dataset("proteinglm/cloning_clf", 
                                sequence_column='seq', 
                                target_columns=self.target_fields, 
                                verbose=verbose)
            
            if 'label' in self.targets:
                self.targets['target'] = [None if v is None else int(v) for v in self.targets.pop('label')]
            else:
                self.targets['target'] = [None] * len(self.sequences)

        except Exception as e:
            logger.exception("Error loading cloning clf dataset")
            raise


def create_dataset(dataset_type, **kwargs):
    datasets = {
        'Thermostability': Thermostability,
        'SecondaryStructure': SecondaryStructure,
        'CloningCLF': CloningCLF
    }
    if dataset_type not in datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(datasets.keys())}")
    return datasets[dataset_type](**kwargs)