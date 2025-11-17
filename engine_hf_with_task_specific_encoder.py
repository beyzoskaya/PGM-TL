import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from protbert_hf import SharedProtBert
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

# ----------------------------
# Utility Metrics
# ----------------------------
def regression_metrics(preds, targets):
    mse = ((preds - targets) ** 2).mean().item()
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse}

def classification_metrics(preds, targets, ignore_index=-100):
    """
    Supports per-sequence or per-residue classification
    preds: [B, num_classes] or [B, L, num_classes]
    targets: [B] or [B, L] with -100 for ignored positions
    """
    preds_flat = preds.view(-1, preds.shape[-1])
    targets_flat = targets.view(-1)
    mask = targets_flat != ignore_index
    if mask.sum() == 0:
        return {'accuracy': float('nan')}
    pred_labels = preds_flat.argmax(dim=-1)
    correct = (pred_labels[mask] == targets_flat[mask]).sum().item()
    acc = correct / mask.sum().item()
    return {'accuracy': acc}

# ----------------------------
# Custom collate_fn for variable-length sequences
# ----------------------------
def collate_fn(batch, tokenizer=None, max_length=None):
    sequences = [item['sequence'] for item in batch]
    targets = [item['targets']['target'] for item in batch]

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided to collate_fn")

    encodings = tokenizer(sequences,
                          return_tensors='pt',
                          padding=True,
                          truncation=True,
                          max_length=max_length)

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Process targets
    if isinstance(targets[0], (int, float)):
        targets = torch.tensor(targets, dtype=torch.float32)
    else:
        # Pad per-residue targets to match input_ids length
        max_len = input_ids.size(1)
        padded_targets = []
        for t in targets:
            t_tensor = torch.tensor(t, dtype=torch.long)
            if len(t_tensor) < max_len:
                pad_len = max_len - len(t_tensor)
                t_tensor = torch.cat([t_tensor, torch.full((pad_len,), -100, dtype=torch.long)])
            else:
                t_tensor = t_tensor[:max_len]
            padded_targets.append(t_tensor)
        targets = torch.stack(padded_targets, dim=0)  # [B, L]

    return {
        'sequence': input_ids,
        'attention_mask': attention_mask,
        'targets': {'target': targets}
    }

# ----------------------------
# MultiTask Engine with Dynamic Weight Logging
# ----------------------------
class MultiTaskEngine:
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets,
                 batch_size=16, device='cuda', alpha=0.99):
        self.device = device
        self.backbone = backbone.to(device)
        self.task_configs = task_configs

        # DataLoaders with custom collate_fn
        self.train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer=self.backbone.tokenizer)) for ds in train_sets]
        self.valid_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer=self.backbone.tokenizer)) for ds in valid_sets]
        self.test_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer=self.backbone.tokenizer)) for ds in test_sets]

        # Task heads
        hidden_dim = backbone.hidden_size
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, cfg['num_labels']) for cfg in task_configs
        ]).to(device)

        # Loss functions
        self.loss_fns = []
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.loss_fns.append(nn.MSELoss())
            elif cfg['type'] in ['classification', 'per_residue_classification']:
                self.loss_fns.append(nn.CrossEntropyLoss(ignore_index=-100))
            else:
                raise ValueError(f"Unknown task type: {cfg['type']}")

        self.running_losses = torch.ones(len(task_configs), device=device)
        self.alpha = alpha
        self.dynamic_weight_log = []

    def forward_task(self, x, task_idx):
        task_type = self.task_configs[task_idx]['type']
        if task_type == 'per_residue_classification':
            print(f"Task {task_idx} - per_residue_classification forward")
            # x: [B, L, D]
            # apply linear token-wise
            B, L, D = x.shape
            logits = self.task_heads[task_idx](x)  # [B, L, num_labels]
            return logits
        else:
            return self.task_heads[task_idx](x)  # [B, D] -> [B, num_labels]


    def compute_loss(self, logits, targets, task_idx):
        task_type = self.task_configs[task_idx]['type']
        if task_type == 'regression':
            return self.loss_fns[task_idx](logits.squeeze(-1), targets)
        elif task_type == 'per_residue_classification':
            # logits: [B, L, C], targets: [B, L]
            return self.loss_fns[task_idx](logits.view(-1, logits.shape[-1]), targets.view(-1))
        else:  # sequence-level classification
            return self.loss_fns[task_idx](logits, targets.long())

    def update_dynamic_weights(self):
        inv_losses = 1.0 / (self.running_losses + 1e-8)
        self.task_weights = inv_losses / inv_losses.sum()
        self.dynamic_weight_log.append(self.task_weights.cpu().numpy())
        return self.task_weights

    def train_one_epoch(self, optimizer):
        self.backbone.train()
        for head in self.task_heads:
            head.train()

        total_loss = 0
        for task_idx, loader in enumerate(self.train_loaders):
            print(f"\n--- Training Task {task_idx} ---")
            for batch_idx, batch in enumerate(loader):
                seqs = batch['sequence'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets']['target']
                if torch.is_tensor(targets):
                    targets = targets.to(self.device)

                optimizer.zero_grad()
                #embeddings = self.backbone(seqs, attention_mask)
                per_residue = self.task_configs[task_idx]['type'] == 'per_residue_classification'
                embeddings = self.backbone(seqs, attention_mask, per_residue=per_residue)
                logits = self.forward_task(embeddings, task_idx)
                loss = self.compute_loss(logits, targets, task_idx)

                self.running_losses[task_idx] = self.alpha * self.running_losses[task_idx] + (1 - self.alpha) * loss.detach()
                weights = self.update_dynamic_weights()
                weighted_loss = weights[task_idx] * loss

                weighted_loss.backward()
                optimizer.step()

                total_loss += weighted_loss.item()
                print(f"Batch {batch_idx}: raw_loss={loss.item():.4f}, weight={weights[task_idx]:.4f}, weighted_loss={weighted_loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loaders)
        print(f"Average weighted training loss: {avg_loss:.4f}")
        self.plot_dynamic_weights()
        return avg_loss

    def evaluate(self, loaders, split_name="Validation"):
        self.backbone.eval()
        for head in self.task_heads:
            head.eval()

        all_metrics = []
        with torch.no_grad():
            for task_idx, loader in enumerate(loaders):
                task_metrics_accum = []
                for batch in loader:
                    seqs = batch['sequence'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets']['target']
                    if torch.is_tensor(targets):
                        targets = targets.to(self.device)

                    #embeddings = self.backbone(seqs, attention_mask)
                    per_residue = self.task_configs[task_idx]['type'] == 'per_residue_classification'
                    embeddings = self.backbone(seqs, attention_mask, per_residue=per_residue)
                    logits = self.forward_task(embeddings, task_idx)

                    if self.task_configs[task_idx]['type'] == 'regression':
                        batch_metrics = regression_metrics(logits.squeeze(-1), targets)
                    else:
                        batch_metrics = classification_metrics(logits, targets, ignore_index=-100)

                    task_metrics_accum.append(batch_metrics)

                # Average metrics over batches
                avg_metrics = {}
                for k in task_metrics_accum[0].keys():
                    avg_metrics[k] = np.mean([m[k] for m in task_metrics_accum])
                print(f"{split_name} metrics for Task {task_idx}: {avg_metrics}")
                all_metrics.append(avg_metrics)
        return all_metrics

    def plot_dynamic_weights(self):
        log = np.array(self.dynamic_weight_log)
        plt.figure(figsize=(8, 4))
        for i in range(log.shape[1]):
            plt.plot(log[:, i], label=f'Task {i} weight')
        plt.xlabel("Batch")
        plt.ylabel("Dynamic Task Weight")
        plt.title("Dynamic Task Weights over Training Batches")
        plt.legend()
        plt.grid(True)
        plt.show()

def validate_dataset_splits(train_set, valid_set, test_set, name="Dataset"):
    print(f"\n=== Checking {name} splits ===")

    # 1. Print sizes
    print(f"Train samples: {len(train_set)}")
    print(f"Valid samples: {len(valid_set)}")
    print(f"Test samples:  {len(test_set)}")

    # 2. Validate they are subsets (expect Subset)
    def get_indices(d):
        if hasattr(d, "indices"):
            return list(d.indices)
        return list(range(len(d)))

    train_idx = set(get_indices(train_set))
    valid_idx = set(get_indices(valid_set))
    test_idx  = set(get_indices(test_set))

    # 3. Confirm disjoint
    inter_train_valid = train_idx & valid_idx
    inter_train_test  = train_idx & test_idx
    inter_valid_test  = valid_idx & test_idx

    # Report any overlaps
    if len(inter_train_valid) == 0 and len(inter_train_test) == 0 and len(inter_valid_test) == 0:
        print("✔ Splits are disjoint (good)")
    else:
        print("❌ Overlap detected!")
        print("train ∩ valid:", inter_train_valid)
        print("train ∩ test:", inter_train_test)
        print("valid ∩ test:", inter_valid_test)

    # 4. Inspect random sample
    import random
    idx = random.randint(0, len(train_set)-1)
    sample = train_set[idx]
    print(f"\nRandom {name} train sample at idx {idx}:")
    print("Sequence (first 40 aa):", sample['sequence'][:40], "...")

    # 5. Print target dict sample
    print("Targets:", sample["targets"])

    # 6. Check missing label count (None labels)
    missing = sum(
        1 for i in range(len(train_set))
        if train_set[i]["targets"]["target"] is None
    )
    print(f"Missing labels in train: {missing}")

    print("=" * 60)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------------
    # Load datasets
    # ----------------------------
    # Thermostability (regression)
    thermo_full = Thermostability()
    thermo_train, thermo_valid, thermo_test = thermo_full.split()

    # Secondary structure (per-residue classification)
    ssp_full = SecondaryStructure()
    ssp_train, ssp_valid, ssp_test = ssp_full.split()

    # Cloning classification (single-label classification)
    clf_full = CloningCLF()
    clf_train, clf_valid, clf_test = clf_full.split()

    print(len(thermo_train), len(thermo_valid), len(thermo_test))
    print(len(ssp_train), len(ssp_valid), len(ssp_test))
    print(len(clf_train), len(clf_valid), len(clf_test))

    validate_dataset_splits(thermo_train, thermo_valid, thermo_test, name="Thermostability")
    validate_dataset_splits(ssp_train, ssp_valid, ssp_test, name="Secondary Structure")
    validate_dataset_splits(clf_train, clf_valid, clf_test, name="Cloning")

    # ----------------------------
    # Task configuration
    # ----------------------------
    task_configs = [
        {'type': 'regression', 'num_labels': 1},
        {'type': 'per_residue_classification', 'num_labels': 8},
        {'type': 'classification', 'num_labels': 2}
    ]

    # ----------------------------
    # Backbone model
    # ----------------------------
    backbone = SharedProtBert(lora=False).to(device)

    # ----------------------------
    # MultiTaskEngine
    # ----------------------------
    engine = MultiTaskEngine(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=[thermo_train, ssp_train, clf_train],
        valid_sets=[thermo_valid, ssp_valid, clf_valid],
        test_sets=[thermo_test, ssp_test, clf_test],
        batch_size=16,
        device=device,
        alpha=0.99
    )

    # ----------------------------
    # Optimizer
    # ----------------------------
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(engine.task_heads.parameters()),
        lr=1e-4
    )

    # ----------------------------
    # Train for one epoch
    # ----------------------------
    print("\n=== Start Training One Epoch with Dynamic Task Weighting ===")
    #engine.train_one_epoch(optimizer)

    # ----------------------------
    # Evaluate on validation sets
    # ----------------------------
    print("\n=== Evaluate on Validation Sets ===")
    #engine.evaluate(engine.valid_loaders, split_name="Validation")

    # ----------------------------
    # Evaluate on test sets
    # ----------------------------
    print("\n=== Evaluate on Test Sets ===")
    #engine.evaluate(engine.test_loaders, split_name="Test")

