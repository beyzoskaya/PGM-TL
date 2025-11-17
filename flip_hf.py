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
    Raw sequences and targets
    No tokenization 
    (tokenization is done in collate_fn so tokenization and label alignment handle together)
    """

    def __init__(self):
        self.sequences = [] # list of raw sequences --> list[str]
        self.targets = {}  # dict of list of targets --> dict[target_name]
        self.num_samples = [0, 0, 0] # number of samples for train/val/test splits
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target_dict = {}
        for key, values in self.targets.items():
            if idx < len(values):
                target_dict[key] = values[idx]
                #print(f"Info: Retrieved target {key} for index {idx}")
            else:
                target_dict[key] = None
                #print(f"Warning: Target {key} missing for index {idx}")
        
        return {
            'sequence': seq,
            'targets': target_dict,
            'graph': self.create_protein_graph(seq)
        }

    def create_protein_graph(self, sequence):
        aa_to_id = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
            'X': 20, 'U': 21, 'B': 22, 'Z': 23, 'O': 24
        }
        residue_type = [aa_to_id.get(aa, 20) for aa in sequence]
        
        return {
            'residue_type': torch.tensor(residue_type, dtype=torch.long),
            'num_residues': torch.tensor([len(residue_type)], dtype=torch.long)
        }
    
    def dataset_summary(self):
        print(f"Sequences: {len(self.sequences)}; Targets: {list(self.targets.keys())}")

class HuggingFaceDataset(ProteinDataset):
    
    def load_hf_dataset(self, dataset_name,
                        split_column='split',
                        sequence_column='sequence',
                        target_columns=None,
                        verbose=1,
                        valid_ratio=0.1,
                        test_ratio=0.1):
        """
        Safe HuggingFace dataset loader
        Guarantees: TRAIN, VALID, TEST always exist
        Never merges splits
        Never returns empty splits silently
        """

        if target_columns is None:
            target_columns = []

        dataset = load_dataset(dataset_name)

        # CASE 1 — standard HF splits exist already
        if isinstance(dataset, dict) and 'train' in dataset:

            train_data = dataset['train']
            valid_data = dataset.get('validation', dataset.get('valid', None))
            test_data  = dataset.get('test', None)

            # ✔ train / valid / test all exist — OK
            if valid_data is not None and test_data is not None:
                if verbose:
                    print(f"✓ Using provided TRAIN / VALID / TEST splits.")

            # ✔ only train & test — create validation from train
            elif valid_data is None and test_data is not None:
                if verbose:
                    print(f"⚠ TRAIN + TEST found — splitting VALIDATION from TRAIN")
                split = train_data.train_test_split(test_size=valid_ratio, seed=42)
                train_data = split['train']
                valid_data = split['test']

            # ✔ only train — create validation and test
            else:
                if verbose:
                    print(f"⚠ Only TRAIN found — creating VALID & TEST splits")

                # first split: train vs temp
                split1 = train_data.train_test_split(test_size=(valid_ratio + test_ratio), seed=42)
                train_data = split1['train']
                temp = split1['test']

                # second split: temp -> valid/test
                relative_test_ratio = test_ratio / (valid_ratio + test_ratio)
                split2 = temp.train_test_split(test_size=relative_test_ratio, seed=42)

                valid_data = split2['train']
                test_data = split2['test']

        else:
            # RARE CASE: only one list exists
            if verbose:
                print(f"⚠ Nonstandard dataset format — manual splitting...")

            all_data = list(dataset.values())[0] if isinstance(dataset, dict) else dataset

            split1 = all_data.train_test_split(test_size=(valid_ratio + test_ratio), seed=42)
            train_data = split1['train']
            temp = split1['test']

            relative_test_ratio = test_ratio / (valid_ratio + test_ratio)
            split2 = temp.train_test_split(test_size=relative_test_ratio, seed=42)
            valid_data = split2['train']
            test_data = split2['test']

        # --- Now we are guaranteed correct splits ---
        splits = [train_data, valid_data, test_data]

        # Collect sequences + targets in order
        all_sequences = []
        all_targets = defaultdict(list)

        for split in splits:
            for item in split:
                seq = item.get(sequence_column) or item.get('seq') or ""
                all_sequences.append(seq)

                # each target key we requested
                for col in target_columns:
                    val = item.get(col, None)
                    # keep None == missing target
                    all_targets[col].append(val)

        # assign unified internal storage
        self.sequences = all_sequences

        for col in target_columns:
            values = all_targets[col]
            self.targets[col] = values

        self.num_samples = [len(train_data), len(valid_data), len(test_data)]

        if verbose:
            print(f"✓ Loaded dataset '{dataset_name}'")
            print(f"  TRAIN: {self.num_samples[0]}")
            print(f"  VALID: {self.num_samples[1]}")
            print(f"  TEST:  {self.num_samples[2]}")
            print(f"  Targets: {target_columns}")

class Thermostability(HuggingFaceDataset):
    """
    Regression dataset wrapper. Stores regression target under 'target' key as floats or None.
    Also computes train-set mean/std for standardization (useful in collate).
    """

    target_fields = ["label"]

    def __init__(self, path=None, split="human_cell", verbose=1, **kwargs):
        super().__init__()
        try:
            dataset_name = "proteinglm/stability_prediction"
            self.load_hf_dataset(dataset_name, sequence_column='seq', target_columns=self.target_fields, verbose=verbose)
            # normalize key name to 'target'
            if 'label' in self.targets:
                self.targets['target'] = [None if v is None else float(v) for v in self.targets.pop('label')]
            else:
                # fallback
                self.targets['target'] = [None] * len(self.sequences)

            # Compute training mean/std from training split if available
            train_n = self.num_samples[0] if len(self.sequences) >= sum(self.num_samples) else self.num_samples[0]
            train_vals = [v for v in self.targets['target'][:self.num_samples[0]] if v is not None]
            if len(train_vals) > 0:
                self.reg_mean = float(np.mean(train_vals))
                self.reg_std = float(np.std(train_vals)) if float(np.std(train_vals)) > 0 else 1.0
            else:
                self.reg_mean = 0.0
                self.reg_std = 1.0

            if verbose:
                print("Thermostability loaded. Example targets:", self.targets['target'][:5])
                print(f"  regression mean/std (train): {self.reg_mean:.4f}/{self.reg_std:.4f}")

        except Exception as e:
            logger.exception("Error loading thermostability dataset")
            raise

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            if num_sample > 0:
                splits.append(Subset(self, range(offset, offset + num_sample)))
                offset += num_sample
            else:
                splits.append(Subset(self, []))
        return splits

class SecondaryStructure(HuggingFaceDataset):
    """
    Token-level classification dataset. Exposes per-residue integer labels under 'target'
    as lists of ints or None. Any invalid labels are coerced to 0 (coil) and missing labels
    set to empty list / None (collate must handle).
    """

    target_fields = ["label"]

    def __init__(self, path=None, verbose=1, **kwargs):
        super().__init__()
        try:
            dataset_name = "proteinglm/ssp_q8"
            self.load_hf_dataset(dataset_name, sequence_column='seq', target_columns=self.target_fields, verbose=verbose)

            # Convert 'label' to list-of-ints stored under 'target'
            processed_targets = []
            invalid_count = 0
            length_mismatch_count = 0

            raw_labels = self.targets.get('label', [None] * len(self.sequences))
            for idx, item in enumerate(raw_labels):
                if item is None:
                    processed_targets.append([])
                    continue

                if isinstance(item, str):
                    # string of digits e.g. '01230123' -> list of ints
                    chars = list(item.strip())
                    vals = []
                    for i, ch in enumerate(chars):
                        try:
                            v = int(ch)
                            if v < 0 or v > 7:
                                invalid_count += 1
                                v = 0
                        except Exception:
                            invalid_count += 1
                            v = 0
                        vals.append(v)
                    # check length
                    seq_len = len(self.sequences[idx]) if idx < len(self.sequences) else 0
                    if len(vals) != seq_len:
                        length_mismatch_count += 1
                        # do not discard: keep len(vals) (will be aligned in collate)
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
                    # unknown format -> empty list
                    processed_targets.append([])

            self.targets['target'] = processed_targets
            # remove old 'label' key if present
            if 'label' in self.targets:
                self.targets.pop('label', None)

            if verbose:
                print("\n" + "=" * 60)
                print("SecondaryStructure Dataset (Q8) summary")
                print(f"  Samples loaded: {len(self.sequences)}")
                example_len = len(self.sequences[0]) if self.sequences else 'N/A'
                print(f"  Example sequence length: {example_len}")
                if self.targets['target']:
                    print(f"  Example target (first 30 residues): {self.targets['target'][0][:30]}")
                print(f"  Invalid labels fixed: {invalid_count}")
                print(f"  Label/sequence length mismatches: {length_mismatch_count}")
                print("=" * 60 + "\n")
        except Exception as e:
            logger.exception("Error loading secondary structure dataset")
            raise

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            if num_sample > 0:
                splits.append(Subset(self, range(offset, offset + num_sample)))
                offset += num_sample
            else:
                splits.append(Subset(self, []))
        return splits

class CloningCLF(HuggingFaceDataset):
    target_fields = ["label"]

    def __init__(self, path=None, verbose=1, **kwargs):
        super().__init__()
        try:
            dataset_name = "proteinglm/cloning_clf"
            self.load_hf_dataset(dataset_name, sequence_column='seq', target_columns=self.target_fields, verbose=verbose)
            if 'label' in self.targets:
                self.targets['target'] = [None if v is None else int(v) for v in self.targets.pop('label')]
            else:
                self.targets['target'] = [None] * len(self.sequences)
            if verbose:
                print(f"CloningCLF loaded. Sample targets: {self.targets['target'][:5]}")
        except Exception as e:
            logger.exception("Error loading cloning clf dataset")
            raise

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            if num_sample > 0:
                splits.append(Subset(self, range(offset, offset + num_sample)))
                offset += num_sample
            else:
                splits.append(Subset(self, []))
        return splits

def create_dataset(dataset_type, **kwargs):
    datasets = {
        'Thermostability': Thermostability,
        'SecondaryStructure': SecondaryStructure,
        'CloningCLF': CloningCLF
    }
    if dataset_type not in datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(datasets.keys())}")
    return datasets[dataset_type](**kwargs)