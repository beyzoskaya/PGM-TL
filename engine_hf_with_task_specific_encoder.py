import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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

        # Tokenizer from backbone
        self.tokenizer = backbone.tokenizer  # assuming backbone has tokenizer

        # Prepare dataloaders with proper collate function
        self.train_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_worker,
                       collate_fn=lambda b: self.tokenize_and_collate(b))
            for ds in train_sets
        ]
        self.valid_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_worker,
                       collate_fn=lambda b: self.tokenize_and_collate(b))
            for ds in valid_sets
        ]
        self.test_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_worker,
                       collate_fn=lambda b: self.tokenize_and_collate(b))
            for ds in test_sets
        ]

    def tokenize_and_collate(self, batch):
        """
        Collate function to tokenize sequences and prepare target tensors.
        """
        sequences = [s['sequence'] for s in batch]
        targets = [s['targets'] for s in batch]

        # Tokenize sequences
        encoding = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Convert targets to tensors
        batch_targets = {}
        for key in targets[0].keys():
            values = [t[key] for t in targets]
            if isinstance(values[0], list):  # per-residue targets
                max_len = max(len(v) for v in values)
                padded = [v + [0]*(max_len-len(v)) for v in values]
                batch_targets[key] = torch.tensor(padded, dtype=torch.long)
            else:
                batch_targets[key] = torch.tensor(values, dtype=torch.float).unsqueeze(-1)

        # Move everything to device
        for k in batch_targets:
            batch_targets[k] = batch_targets[k].to(self.device)
        for k in encoding:
            encoding[k] = encoding[k].to(self.device)

        # Debug prints for first batch only
        if hasattr(self, '_debug_done') is False:
            print("=== Debug collate ===")
            print("Input IDs shape:", encoding['input_ids'].shape)
            for k, v in batch_targets.items():
                print(f"Target '{k}' shape: {v.shape}")
            self._debug_done = True

        return encoding, batch_targets

    # Optional: keep old print_first_batch for debugging
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
        encoding, targets = batch
        print(f"=== First batch from {split} dataset {dataset_idx} ===")
        print("Input IDs shape:", encoding['input_ids'].shape)
        for k, v in targets.items():
            print(f"Target '{k}' shape: {v.shape}")
        return batch
    
    def forward(self, batch, dataset_idx=0):
        """
        Forward pass through the backbone for a given batch.
        batch: (encoding, targets) returned by tokenize_and_collate
        dataset_idx: which dataset/task this batch belongs to
        """
        encoding, targets = batch
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Forward through backbone
        embeddings = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Debug prints
        print(f"=== Forward pass for dataset {dataset_idx} ===")
        print("Input IDs shape:", input_ids.shape)
        print("Embeddings shape:", embeddings.shape)
        for k, v in targets.items():
            print(f"Target '{k}' shape: {v.shape}")

        return embeddings, targets

