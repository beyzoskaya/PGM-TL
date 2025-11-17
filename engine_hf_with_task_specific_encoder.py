import torch
from torch.utils.data import DataLoader

class MultiTaskEngine:
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets,
                 batch_size=8, num_worker=0, device='cpu'):
        """
        backbone: SharedProtBert model
        task_configs: list of dicts, e.g. [{'type':'regression', 'num_labels':1}, ...]
        train_sets, valid_sets, test_sets: list of Dataset objects
        """
        self.backbone = backbone
        self.task_configs = task_configs
        self.device = device
        self.batch_size = batch_size
        self.num_worker = num_worker

        # Datasets
        self.train_sets = train_sets
        self.valid_sets = valid_sets
        self.test_sets = test_sets

        # Prepare dataloaders
        self.train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         num_workers=num_worker, collate_fn=self.collate_fn)
                              for ds in train_sets]
        self.valid_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=num_worker, collate_fn=self.collate_fn)
                              for ds in valid_sets]
        self.test_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False,
                                        num_workers=num_worker, collate_fn=self.collate_fn)
                             for ds in test_sets]

    def collate_fn(self, batch):
        """Temporary collate_fn: just pass sequences and targets as lists"""
        sequences = [sample['sequence'] for sample in batch]
        targets = [sample['targets'] for sample in batch]
        return {'sequences': sequences, 'targets': targets}

    def print_first_batch(self, split='train', dataset_idx=0):
        if split == 'train':
            loader = self.train_loaders[dataset_idx]
        elif split == 'valid':
            loader = self.valid_loaders[dataset_idx]
        elif split == 'test':
            loader = self.test_loaders[dataset_idx]
        else:
            raise ValueError("split must be 'train', 'valid', or 'test'")

        batch = next(iter(loader))
        print(f"=== First batch from {split} dataset {dataset_idx} ===")
        for i, (seq, targ) in enumerate(zip(batch['sequences'], batch['targets'])):
            print(f"Sample {i}: sequence length={len(seq)}, target keys={list(targ.keys())}")
            for k, v in targ.items():
                if isinstance(v, list):
                    print(f"  Target '{k}' (first 10): {v[:10]}")
                else:
                    print(f"  Target '{k}': {v}")
        return batch
