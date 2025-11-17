import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiTaskEngine:
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets,
                 batch_size=8, num_worker=0, device=device):
        self.backbone = backbone
        self.task_configs = task_configs
        self.device = device
        self.batch_size = batch_size
        self.num_worker = num_worker

        # Datasets
        self.train_sets = train_sets
        self.valid_sets = valid_sets
        self.test_sets = test_sets

        # Tokenizer
        if hasattr(backbone, 'tokenizer'):
            self.tokenizer = backbone.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)

        # Task-specific heads
        self.task_heads = nn.ModuleList()
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.task_heads.append(nn.Linear(1024, 1))
            elif cfg['type'] in ['classification', 'per_residue_classification']:
                self.task_heads.append(nn.Linear(1024, cfg['num_labels']))
            else:
                raise ValueError(f"Unknown task type: {cfg['type']}")
        self.task_heads.to(device)

        # DataLoaders
        self.train_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_worker,
                       collate_fn=lambda b: self.tokenize_and_collate(b))
            for ds in train_sets
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

        # Move to device
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
        encoding, targets = batch
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        task_cfg = self.task_configs[dataset_idx]

        if task_cfg['type'] == 'per_residue_classification':
            # Full sequence embeddings
            outputs = self.backbone.backbone.bert(input_ids=input_ids,
                                                  attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # [B, L, D]

            # Pad/truncate embeddings to match target length
            target_tensor = list(targets.values())[0]
            target_len = target_tensor.shape[1]

            if embeddings.shape[1] != target_len:
                pad_size = target_len - embeddings.shape[1]
                if pad_size > 0:
                    pad_tensor = torch.zeros(embeddings.shape[0], pad_size, embeddings.shape[2], device=embeddings.device)
                    embeddings = torch.cat([embeddings, pad_tensor], dim=1)
                else:
                    embeddings = embeddings[:, :target_len, :]
        else:
            # Pooled embeddings
            embeddings = self.backbone(input_ids=input_ids, attention_mask=attention_mask)  # [B, D]

        logits = self.task_heads[dataset_idx](embeddings)
        return logits, targets

    def train_one_epoch(self, optimizer, max_batches_per_task=None):
        """
        Single epoch over all tasks, one task at a time.
        """
        self.backbone.train()
        for task_idx, loader in enumerate(self.train_loaders):
            task_cfg = self.task_configs[task_idx]

            # Loss function
            if task_cfg['type'] == 'regression':
                loss_fn = nn.MSELoss()
            elif task_cfg['type'] in ['classification', 'per_residue_classification']:
                loss_fn = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown task type: {task_cfg['type']}")

            print(f"\n--- Training task {task_idx} ({task_cfg['type']}) ---")
            for batch_idx, batch in enumerate(loader):
                if max_batches_per_task is not None and batch_idx >= max_batches_per_task:
                    break

                optimizer.zero_grad()
                logits, targets = self.forward(batch, dataset_idx=task_idx)
                target_tensor = list(targets.values())[0]

                # Compute loss
                if task_cfg['type'] == 'regression':
                    loss = loss_fn(logits, target_tensor)
                elif task_cfg['type'] == 'classification':
                    loss = loss_fn(logits, target_tensor.squeeze(-1).long())
                elif task_cfg['type'] == 'per_residue_classification':
                    if logits.shape[:2] != target_tensor.shape[:2]:
                        raise ValueError(f"Logits shape {logits.shape} != target shape {target_tensor.shape}")
                    b, s, c = logits.shape
                    loss = loss_fn(logits.view(b*s, c), target_tensor.view(b*s))

                # Backprop
                loss.backward()
                optimizer.step()

                if batch_idx == 0:
                    print(f"Batch {batch_idx}: loss={loss.item()}")
                    print("Logits shape:", logits.shape)
                    print("Target shape:", target_tensor.shape)
