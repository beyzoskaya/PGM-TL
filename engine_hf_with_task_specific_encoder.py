import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import os
from itertools import cycle

from protbert_hf import SharedProtBert, build_regression_head, build_token_classification_head, build_sequence_classification_head

MAX_LENGTH = 512
DEBUG_INTERVAL = 100  # Log embeddings/logits every 100 batches

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def regression_metrics(preds, targets):
    preds = preds.squeeze(-1)
    targets = targets.squeeze(-1)
    mse = ((preds - targets) ** 2).mean().item()
    rmse = float(np.sqrt(mse))
    mae = (preds - targets).abs().mean().item()
    return {'mse': mse, 'rmse': rmse, 'mae': mae}

def classification_metrics(preds, targets, ignore_index=-100):
    if preds.dim() == 3:  # per-residue
        B, L, C = preds.shape
        preds_flat = preds.view(B*L, C)
        targets_flat = targets.view(B*L)
        mask = targets_flat != ignore_index
        if mask.sum() == 0:
            return {'accuracy': float('nan')}
        pred_labels = preds_flat.argmax(dim=-1)
        correct = (pred_labels[mask] == targets_flat[mask]).sum().item()
        acc = correct / mask.sum().item()
        return {'accuracy': acc}
    else:  # sequence classification
        pred_labels = preds.argmax(dim=-1)
        targets_flat = targets.view(-1)
        acc = (pred_labels == targets_flat).sum().item() / targets_flat.numel()
        return {'accuracy': acc}

def collate_fn(batch, tokenizer, max_length=MAX_LENGTH):
    sequences = [item['sequence'] for item in batch]
    targets = [item['targets']['target'] for item in batch]

    encodings = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Process targets
    if isinstance(targets[0], (int, float)):
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)  # Ensure [B,1]
    else:
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
        targets = torch.stack(padded_targets, dim=0)

    return {
        'sequence': input_ids,
        'attention_mask': attention_mask,
        'targets': {'target': targets}
    }

def debug_embeddings(task_idx, embeddings):
    emean = embeddings.mean().item()
    emax = embeddings.max().item()
    emin = embeddings.min().item()
    print(f"[EMB DEBUG][Task {task_idx}] shape={tuple(embeddings.shape)}, mean={emean:.4f}, min={emin:.4f}, max={emax:.4f}")

def debug_logits(task_idx, logits):
    mean = logits.mean().item()
    mx = logits.max().item()
    mn = logits.min().item()
    print(f"[LOGIT DEBUG][Task {task_idx}] shape={tuple(logits.shape)}, mean={mean:.4f}, min={mn:.4f}, max={mx:.4f}")

def debug_grad_norm(model):
    total_grad_norm = 0.0
    num_params = 0
    zero_grads = 0
    for n, p in model.named_parameters():
        if p.grad is not None:
            gnorm = p.grad.norm().item()
            total_grad_norm += gnorm ** 2
            num_params += 1
            if gnorm == 0:
                zero_grads += 1
    total_grad_norm = float(np.sqrt(total_grad_norm))
    return total_grad_norm, num_params, zero_grads

class MultiTaskEngine(nn.Module):
    """
    Multi-task engine with uncertainty weighting (log_vars) for task losses.
    Supports regression, per-residue classification, and sequence classification.
    """
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets,
                 batch_size=16, device='cuda', save_dir=None, max_length=MAX_LENGTH,
                 verbose=True, grad_clip=1.0):
        super().__init__()
        self.device = device
        self.backbone = backbone.to(device)
        self.task_configs = task_configs
        self.num_tasks = len(task_configs)
        self.max_length = max_length
        self.verbose = verbose
        self.grad_clip = grad_clip

        tokenizer = self.backbone.tokenizer
        self.train_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True,
                       collate_fn=lambda b, tok=tokenizer: collate_fn(b, tok, max_length=max_length))
            for ds in train_sets
        ]
        self.valid_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False,
                       collate_fn=lambda b, tok=tokenizer: collate_fn(b, tok, max_length=max_length))
            for ds in valid_sets
        ]
        self.test_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False,
                       collate_fn=lambda b, tok=tokenizer: collate_fn(b, tok, max_length=max_length))
            for ds in test_sets
        ]

        hidden_dim = backbone.hidden_size
        self.task_heads = nn.ModuleList()
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.task_heads.append(build_regression_head(hidden_dim, cfg['num_labels']).to(device))
            elif cfg['type'] == 'per_residue_classification':
                self.task_heads.append(build_token_classification_head(hidden_dim, cfg['num_labels']).to(device))
            else:
                self.task_heads.append(build_sequence_classification_head(hidden_dim, cfg['num_labels']).to(device))

        # Loss functions
        self.loss_fns = []
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.loss_fns.append(nn.MSELoss())
            elif cfg['type'] == 'per_residue_classification':
                self.loss_fns.append(nn.CrossEntropyLoss(ignore_index=-100))
            else:
                self.loss_fns.append(nn.CrossEntropyLoss())

        # Log vars for uncertainty weighting
        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks, device=device))

        # History & bookkeeping
        self.history = {"train_loss": [], "val_metrics": [], "test_metrics": [],
                        "log_vars": [], "gradient_norms": []}
        self.best = {"scores": [None]*self.num_tasks, "epochs": [None]*self.num_tasks}
        self.save_dir = ensure_dir(save_dir) if save_dir else None

    def forward_task(self, embeddings, task_idx):
        return self.task_heads[task_idx](embeddings)

    def compute_task_loss(self, logits, targets, task_idx):
        cfg = self.task_configs[task_idx]
        if cfg['type'] == 'per_residue_classification':
            B, L, C = logits.shape
            logits_flat = logits.view(B*L, C)
            targets_flat = targets.view(B*L)
            return self.loss_fns[task_idx](logits_flat, targets_flat)
        elif cfg['type'] == 'regression':
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)  # Ensure [B,1]
            return self.loss_fns[task_idx](logits, targets)
        else:
            return self.loss_fns[task_idx](logits, targets)

    def compute_total_loss(self, logits_list, targets_list):
        total_loss = 0.0
        per_task_losses = []
        for i, (logits, targets) in enumerate(zip(logits_list, targets_list)):
            task_loss = self.compute_task_loss(logits, targets, i)
            per_task_losses.append(task_loss.detach().cpu().item())
            weighted_loss = (0.5 * torch.exp(-self.log_vars[i]) * task_loss + 0.5 * self.log_vars[i])
            total_loss += weighted_loss
        return total_loss, per_task_losses

    def train_one_epoch(self, optimizer, max_batches_per_loader=None):
        self.backbone.train()
        for head in self.task_heads:
            head.train()

        iterators = [iter(loader) for loader in self.train_loaders]
        task_cycle = cycle(range(self.num_tasks))
        active_iterators = list(range(self.num_tasks))
        total_loss = 0.0
        updates = 0

        while len(active_iterators) > 0:
            task_idx = next(task_cycle)
            if task_idx not in active_iterators:
                continue
            try:
                batch = next(iterators[task_idx])
            except StopIteration:
                active_iterators.remove(task_idx)
                continue

            input_ids = batch['sequence'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['targets']['target']
            if torch.is_tensor(targets):
                targets = targets.to(self.device)

            optimizer.zero_grad()
            per_residue = (self.task_configs[task_idx]['type'] == 'per_residue_classification')
            embeddings = self.backbone(input_ids, attention_mask, per_residue=per_residue)
            logits = self.forward_task(embeddings, task_idx)

            total_batch_loss, _ = self.compute_total_loss([logits], [targets])
            total_batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.backbone.parameters()) + list(self.task_heads.parameters()),
                max_norm=self.grad_clip
            )
            optimizer.step()

            total_loss += total_batch_loss.item()
            updates += 1

            if updates % DEBUG_INTERVAL == 0:
                grad_norm, num_params, zero_grads = debug_grad_norm(self.backbone)
                print(f"[DEBUG][Batch {updates}] Grad norm={grad_norm:.4e}, Zero grads={zero_grads}")

            if max_batches_per_loader is not None and updates >= max_batches_per_loader * self.num_tasks:
                break

        avg_loss = total_loss / max(1, updates)
        self.history["gradient_norms"].append(debug_grad_norm(self.backbone))
        return avg_loss

    def evaluate(self, loaders, split_name="Validation", epoch=None):
        self.backbone.eval()
        for head in self.task_heads:
            head.eval()
        all_metrics = []

        with torch.no_grad():
            for task_idx, loader in enumerate(loaders):
                acc = []
                for batch in loader:
                    input_ids = batch['sequence'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets']['target']
                    if torch.is_tensor(targets):
                        targets = targets.to(self.device)

                    per_residue = (self.task_configs[task_idx]['type'] == 'per_residue_classification')
                    embeddings = self.backbone(input_ids, attention_mask, per_residue=per_residue)
                    logits = self.forward_task(embeddings, task_idx)

                    if self.task_configs[task_idx]['type'] == 'regression':
                        metrics = regression_metrics(logits.cpu(), targets.cpu())
                    else:
                        metrics = classification_metrics(logits.cpu(), targets.cpu())
                    acc.append(metrics)

                avg_metrics = {k: float(np.mean([a[k] for a in acc])) for k in acc[0].keys()}
                all_metrics.append(avg_metrics)

                if epoch is not None:
                    if self.task_configs[task_idx]['type'] == 'regression':
                        current_score = avg_metrics['rmse']
                        better = self.best["scores"][task_idx] is None or current_score < self.best["scores"][task_idx]
                    else:
                        current_score = avg_metrics['accuracy']
                        better = self.best["scores"][task_idx] is None or current_score > self.best["scores"][task_idx]
                    if better:
                        self.best["scores"][task_idx] = current_score
                        self.best["epochs"][task_idx] = epoch

                if self.verbose:
                    print(f"[{split_name}] Task {task_idx}: {avg_metrics}")

        return all_metrics

    def print_best_metrics(self):
        print("\n" + "="*60)
        print("BEST VALIDATION METRICS PER TASK")
        print("="*60)
        for task_idx in range(self.num_tasks):
            score = self.best["scores"][task_idx]
            epoch = self.best["epochs"][task_idx]
            metric = "RMSE" if self.task_configs[task_idx]['type']=="regression" else "Accuracy"
            if score is not None:
                print(f"Task {task_idx}: {metric}={score:.4f} @ epoch {epoch}")
        print("="*60 + "\n")
