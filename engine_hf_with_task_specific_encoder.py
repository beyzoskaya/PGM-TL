import torch
from torch import nn
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
        if hasattr(backbone, 'tokenizer'):
            self.tokenizer = backbone.tokenizer
        else:
            # fallback: create tokenizer manually if backbone doesn't store it
            self.tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)

        # Task-specific heads
        self.task_heads = nn.ModuleList()
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.task_heads.append(nn.Linear(1024, 1))
            elif cfg['type'] == 'classification':
                self.task_heads.append(nn.Linear(1024, cfg['num_labels']))
            elif cfg['type'] == 'per_residue_classification':
                self.task_heads.append(nn.Linear(1024, cfg['num_labels']))
            else:
                raise ValueError(f"Unknown task type: {cfg['type']}")

        self.task_heads.to(device)

        # Prepare dataloaders
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
        sequences = [s['sequence'] for s in batch]
        targets = [s['targets'] for s in batch]

        encoding = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        batch_targets = {}
        for key in targets[0].keys():
            values = [t[key] for t in targets]
            if isinstance(values[0], list):  # per-residue targets
                max_len = max(len(v) for v in values)
                padded = [v + [0]*(max_len-len(v)) for v in values]
                batch_targets[key] = torch.tensor(padded, dtype=torch.long)
            else:
                batch_targets[key] = torch.tensor(values, dtype=torch.float).unsqueeze(-1)

        for k in batch_targets:
            batch_targets[k] = batch_targets[k].to(self.device)
        for k in encoding:
            encoding[k] = encoding[k].to(self.device)

        # Debug prints for first batch only
        if not hasattr(self, '_debug_done'):
            print("=== Debug collate ===")
            print("Input IDs shape:", encoding['input_ids'].shape)
            for k, v in batch_targets.items():
                print(f"Target '{k}' shape: {v.shape}")
            self._debug_done = True

        return encoding, batch_targets

    def forward(self, batch, dataset_idx=0):
        """
        Forward pass through backbone + task-specific head
        """
        encoding, targets = batch
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Backbone embeddings
        embeddings = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Task-specific head
        task_head = self.task_heads[dataset_idx]
        task_cfg = self.task_configs[dataset_idx]
        if task_cfg['type'] == 'per_residue_classification':
            # Apply head per residue
            # embeddings shape: [batch_size, seq_len, 1024]
            logits = task_head(embeddings)
        else:
            # Apply head on pooled embedding
            # embeddings shape: [batch_size, 1024]
            logits = task_head(embeddings)

        # Debug prints
        print(f"=== Forward pass for dataset {dataset_idx} ===")
        print("Input IDs shape:", input_ids.shape)
        print("Embeddings shape:", embeddings.shape)
        print("Logits shape:", logits.shape)
        for k, v in targets.items():
            print(f"Target '{k}' shape: {v.shape}")

        return logits, targets
