import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiTaskEngine:
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets,
                 batch_size=8, num_worker=0, device=device, reg_loss_scale=1.0):
        self.backbone = backbone
        self.task_configs = task_configs
        self.device = device
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.reg_loss_scale = reg_loss_scale  # scaling factor for regression

        self.train_sets = train_sets
        self.valid_sets = valid_sets
        self.test_sets = test_sets

        if hasattr(backbone, 'tokenizer'):
            self.tokenizer = backbone.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)

        # Task heads
        self.task_heads = nn.ModuleList()
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.task_heads.append(nn.Linear(1024, 1))
            elif cfg['type'] in ['classification', 'per_residue_classification']:
                self.task_heads.append(nn.Linear(1024, cfg['num_labels']))
            else:
                raise ValueError(f"Unknown task type: {cfg['type']}")
        self.task_heads.to(device)

        # Compute regression mean/std for normalization
        self.regression_stats = {}
        for i, cfg in enumerate(task_configs):
            if cfg['type'] == 'regression':
                all_targets = []
                for ds in train_sets[i]:
                    all_targets.append(ds['targets'][cfg.get('target_key','target')])
                all_targets = torch.tensor(all_targets, dtype=torch.float)
                self.regression_stats[i] = {
                    'mean': all_targets.mean().item(),
                    'std': all_targets.std().item()
                }

        # DataLoaders
        self.train_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_worker,
                       collate_fn=lambda b: self.tokenize_and_collate(b, task_idx=i))
            for i, ds in enumerate(train_sets)
        ]

    def tokenize_and_collate(self, batch, task_idx=None):
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
            if isinstance(values[0], list):
                max_len = max(len(v) for v in values)
                padded = [v + [0]*(max_len-len(v)) for v in values]
                batch_targets[key] = torch.tensor(padded, dtype=torch.long)
            else:
                y = torch.tensor(values, dtype=torch.float).unsqueeze(-1)
                # normalize regression targets
                if task_idx in self.regression_stats:
                    mean = self.regression_stats[task_idx]['mean']
                    std = self.regression_stats[task_idx]['std']
                    y = (y - mean) / (std + 1e-8)
                batch_targets[key] = y

        for k in batch_targets:
            batch_targets[k] = batch_targets[k].to(self.device)
        for k in encoding:
            encoding[k] = encoding[k].to(self.device)

        return encoding, batch_targets

    def train_one_epoch(self, optimizer, max_batches_per_task=None):
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
                    loss = loss_fn(logits, target_tensor) * self.reg_loss_scale
                elif task_cfg['type'] == 'classification':
                    loss = loss_fn(logits, target_tensor.squeeze(-1).long())
                elif task_cfg['type'] == 'per_residue_classification':
                    b, s, c = logits.shape
                    loss = loss_fn(logits.view(b*s, c), target_tensor.view(b*s))

                loss.backward()
                optimizer.step()

                if batch_idx == 0:
                    print(f"Batch {batch_idx}: loss={loss.item()}")
                    print("Logits shape:", logits.shape)
                    print("Target shape:", target_tensor.shape)
