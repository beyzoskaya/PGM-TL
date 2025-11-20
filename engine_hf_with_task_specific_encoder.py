import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import os
from itertools import cycle

from protbert_hf import SharedProtBert, build_regression_head, build_token_classification_head, build_sequence_classification_head

MAX_LENGTH = 512
DEBUG_INTERVAL = 50

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def regression_metrics(preds, targets):
    """Calculate regression metrics"""
    preds = preds.squeeze(-1)
    targets = targets.squeeze(-1)
    mse = ((preds - targets) ** 2).mean().item()
    rmse = float(np.sqrt(mse))
    mae = (preds - targets).abs().mean().item()
    return {'mse': mse, 'rmse': rmse, 'mae': mae}

def classification_metrics(preds, targets, ignore_index=-100):
    """Calculate classification metrics"""
    if preds.dim() == 3:  # per-residue: [B, L, C]
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
    else:  # sequence-level: [B, C]
        pred_labels = preds.argmax(dim=-1)
        targets_flat = targets.view(-1)
        acc = (pred_labels == targets_flat).sum().item() / targets_flat.numel()
        return {'accuracy': acc}

def collate_fn(batch, tokenizer, max_length=MAX_LENGTH):
    """Collate batch for tokenization and padding"""
    sequences = [item['sequence'] for item in batch]
    targets = [item['targets']['target'] for item in batch]

    encodings = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Process targets based on type
    if isinstance(targets[0], (int, float)):
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    else:
        max_len = input_ids.size(1)
        padded_targets = []
        for t in targets:
            t_tensor = torch.tensor(t, dtype=torch.long) if isinstance(t, list) else torch.tensor([t], dtype=torch.long)
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

def debug_grad_norm(model):
    """Debug gradient norms"""
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
    Multi-task learning engine with CYCLIC task batch processing.
    Handles imbalanced dataset sizes by cycling through shorter dataloaders.
    """
    
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets,
                 batch_size=16, device='cuda', save_dir=None, max_length=MAX_LENGTH,
                 verbose=True, grad_clip=1.0, task_weights=None):
        super().__init__()
        self.device = device
        self.backbone = backbone.to(device)
        self.task_configs = task_configs
        self.num_tasks = len(task_configs)
        self.max_length = max_length
        self.verbose = verbose
        self.grad_clip = grad_clip

        # Task weights: Equal by default (loss normalization handles magnitude difference)
        if task_weights is None:
            self.task_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        else:
            self.task_weights = torch.tensor(task_weights, dtype=torch.float32, device=device)

        tokenizer = self.backbone.tokenizer
        
        # Create data loaders
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
        
        # Build task-specific heads
        self.task_heads = nn.ModuleList()
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.task_heads.append(build_regression_head(hidden_dim, cfg['num_labels']).to(device))
            elif cfg['type'] == 'per_residue_classification':
                self.task_heads.append(build_token_classification_head(hidden_dim, cfg['num_labels']).to(device))
            else:  # sequence_classification
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

        # History
        self.history = {
            "train_loss": [], 
            "val_metrics": [], 
            "test_metrics": [],
            "per_task_losses": [],
            "per_task_grad_norms": []
        }
        self.best = {"scores": [None]*self.num_tasks, "epochs": [None]*self.num_tasks}
        self.save_dir = ensure_dir(save_dir) if save_dir else None
        
        if self.verbose:
            loader_sizes = [len(loader) for loader in self.train_loaders]
            print(f"[Engine] Train loader sizes (batches): {loader_sizes}")
            print(f"[Engine] Task weights: {[f'{w:.2f}' for w in self.task_weights.cpu().numpy()]}")
            print(f"[Engine] Task names: {[cfg['name'] for cfg in task_configs]}")
            print(f"[Engine] Epoch length will be: {max(loader_sizes)} batches × {self.num_tasks} tasks = {max(loader_sizes) * self.num_tasks} total updates")
            print(f"[Engine] Shorter tasks will cycle, e.g.: Task(batch 1→2→...→{loader_sizes[0]}) CYCLE back to 1, then 2...")

    def forward_task(self, embeddings, task_idx):
        """Forward pass through task head"""
        return self.task_heads[task_idx](embeddings)

    def compute_task_loss(self, logits, targets, task_idx):
        """Compute loss for a single task"""
        cfg = self.task_configs[task_idx]
        
        if cfg['type'] == 'regression':
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
            logits = logits.squeeze(-1) if logits.dim() == 2 and logits.size(-1) == 1 else logits
            return self.loss_fns[task_idx](logits, targets.squeeze(-1))
            
        elif cfg['type'] == 'per_residue_classification':
            B, L, C = logits.shape
            logits_flat = logits.view(B*L, C)
            targets_flat = targets.view(B*L)
            return self.loss_fns[task_idx](logits_flat, targets_flat)
            
        else:  # sequence_classification
            targets_flat = targets.view(-1)
            return self.loss_fns[task_idx](logits, targets_flat.long())

    def compute_weighted_loss(self, task_idx, logits, targets):
        """
        Compute weighted loss with normalization.
        CRITICAL: Per-residue tasks have loss summed over sequence length,
        so we normalize by number of valid tokens to make losses comparable.
        """
        cfg = self.task_configs[task_idx]
        task_loss = self.compute_task_loss(logits, targets, task_idx)
        
        # Normalize for per-residue tasks: loss is summed, convert to per-token average
        if cfg['type'] == 'per_residue_classification':
            num_valid_tokens = (targets != -100).sum().item()
            if num_valid_tokens > 0:
                normalized_loss = task_loss / num_valid_tokens
            else:
                normalized_loss = task_loss
        else:
            # Regression and sequence classification already per-sample
            normalized_loss = task_loss
        
        # Apply task weight (should be ~1.0 for all tasks if losses are normalized)
        weighted_loss = self.task_weights[task_idx] * normalized_loss
        return weighted_loss, task_loss.detach().item(), normalized_loss.detach().item()

    def train_one_epoch(self, optimizer):
        """
        Train for one epoch using CYCLIC batch strategy.
        
        Key innovation: Instead of processing all batches of Task 0, then Task 1, then Task 2,
        we cycle through: batch from Task 0, batch from Task 1, batch from Task 2, repeat.
        
        This ensures:
        1. Balanced gradient updates (each task updates encoder equally)
        2. Shorter dataloaders cycle seamlessly
        3. Encoder learns shared representation, not task-specific one
        """
        self.backbone.train()
        for head in self.task_heads:
            head.train()

        # Create infinite cycle iterators for each task's dataloader
        # These cycle back to start when exhausted
        loaders_cycle = [cycle(loader) for loader in self.train_loaders]
        
        # Epoch length = max batches across all tasks
        # This ensures longest task is fully processed, others cycle
        max_batches = max(len(loader) for loader in self.train_loaders)
        
        total_weighted_loss = 0.0
        per_task_stats = {
            'weighted_loss': [0.0] * self.num_tasks,
            'raw_loss': [0.0] * self.num_tasks,
            'normalized_loss': [0.0] * self.num_tasks,
            'grad_norm': [0.0] * self.num_tasks,
            'updates': [0] * self.num_tasks
        }
        
        if self.verbose:
            loader_sizes = [len(loader) for loader in self.train_loaders]
            print(f"\n[Epoch] Cyclic training with max_batches={max_batches}")
            print(f"        Loader sizes: {loader_sizes}")

        total_updates = 0
        
        for batch_idx in range(max_batches):
            for task_idx in range(self.num_tasks):
                # Get next batch from this task (cycles if needed)
                batch = next(loaders_cycle[task_idx])
                
                input_ids = batch['sequence'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets']['target'].to(self.device)

                optimizer.zero_grad()
                
                # Forward pass through encoder and task head
                per_residue = (self.task_configs[task_idx]['type'] == 'per_residue_classification')
                embeddings = self.backbone(input_ids, attention_mask, per_residue=per_residue)
                logits = self.forward_task(embeddings, task_idx)

                # Compute weighted loss with normalization
                weighted_loss, raw_loss, normalized_loss = self.compute_weighted_loss(task_idx, logits, targets)
                
                # Backward pass
                weighted_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.backbone.parameters()) + list(self.task_heads.parameters()),
                    max_norm=self.grad_clip
                )
                
                # Track gradient norm before step
                grad_norm, _, _ = debug_grad_norm(self.backbone)
                
                # Optimizer step
                optimizer.step()

                # Accumulate statistics
                total_weighted_loss += weighted_loss.item()
                per_task_stats['weighted_loss'][task_idx] += weighted_loss.item()
                per_task_stats['raw_loss'][task_idx] += raw_loss
                per_task_stats['normalized_loss'][task_idx] += normalized_loss
                per_task_stats['grad_norm'][task_idx] += grad_norm
                per_task_stats['updates'][task_idx] += 1
                total_updates += 1

                # Detailed logging every DEBUG_INTERVAL updates
                if total_updates % DEBUG_INTERVAL == 0 and self.verbose:
                    print(f"[Update {total_updates}] Task {task_idx} ({self.task_configs[task_idx]['name']}):")
                    print(f"  Raw loss: {raw_loss:.4f} | Normalized: {normalized_loss:.4f} | Weighted: {weighted_loss.item():.4f}")
                    print(f"  Grad norm: {grad_norm:.4e}")

        # Compute epoch averages
        avg_weighted_loss = total_weighted_loss / max(1, total_updates)
        avg_per_task_stats = {}
        for key in ['weighted_loss', 'raw_loss', 'normalized_loss', 'grad_norm']:
            avg_per_task_stats[key] = [
                per_task_stats[key][i] / max(1, per_task_stats['updates'][i]) 
                for i in range(self.num_tasks)
            ]
        
        # Store history
        self.history["per_task_losses"].append({
            "raw": avg_per_task_stats['raw_loss'],
            "normalized": avg_per_task_stats['normalized_loss'],
            "weighted": avg_per_task_stats['weighted_loss'],
            "updates": per_task_stats['updates']
        })
        self.history["per_task_grad_norms"].append(avg_per_task_stats['grad_norm'])
        
        if self.verbose:
            print(f"\n[Epoch Summary]")
            print(f"  Total updates: {total_updates}")
            print(f"  Updates per task: {per_task_stats['updates']}")
            for i in range(self.num_tasks):
                print(f"  Task {i} ({self.task_configs[i]['name']}):")
                print(f"    Raw loss: {avg_per_task_stats['raw_loss'][i]:.4f}")
                print(f"    Normalized loss: {avg_per_task_stats['normalized_loss'][i]:.4f}")
                print(f"    Weighted loss: {avg_per_task_stats['weighted_loss'][i]:.4f}")
                print(f"    Avg grad norm: {avg_per_task_stats['grad_norm'][i]:.4e}")
        
        return avg_weighted_loss

    def evaluate(self, loaders, split_name="Validation", epoch=None):
        """Evaluate on all tasks"""
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
                    targets = batch['targets']['target'].to(self.device)

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
                    print(f"  [{split_name}] Task {task_idx} ({self.task_configs[task_idx]['name']}): {avg_metrics}")

        return all_metrics

    def print_best_metrics(self):
        print("\n" + "="*70)
        print("BEST VALIDATION METRICS PER TASK")
        print("="*70)
        for task_idx in range(self.num_tasks):
            score = self.best["scores"][task_idx]
            epoch = self.best["epochs"][task_idx]
            metric = "RMSE" if self.task_configs[task_idx]['type'] == "regression" else "Accuracy"
            name = self.task_configs[task_idx]['name']
            if score is not None:
                print(f"Task {task_idx} ({name:25s}): {metric}={score:.4f} @ epoch {epoch}")
            else:
                print(f"Task {task_idx} ({name:25s}): No improvement")
        print("="*70 + "\n")