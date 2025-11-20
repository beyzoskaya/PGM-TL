import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
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
    preds = preds.squeeze(-1)
    targets = targets.squeeze(-1)
    mse = ((preds - targets) ** 2).mean().item()
    rmse = float(np.sqrt(mse))
    mae = (preds - targets).abs().mean().item()
    return {'mse': mse, 'rmse': rmse, 'mae': mae}

def classification_metrics(preds, targets, ignore_index=-100):
    if preds.dim() == 3:
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
    else:
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

def extract_gradient_vector(model):
    """Extract all gradients as a single vector"""
    grad_list = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_list.append(param.grad.flatten())
    if grad_list:
        return torch.cat(grad_list)
    else:
        return torch.zeros(1, device=model.parameters().__next__().device)

class MultiTaskEngine(nn.Module):
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets,
                 batch_size=16, device='cuda', save_dir=None, max_length=MAX_LENGTH,
                 verbose=True, grad_clip=1.0, task_weights=None, check_gradient_conflict=True):
        super().__init__()
        self.device = device
        self.backbone = backbone.to(device)
        self.task_configs = task_configs
        self.num_tasks = len(task_configs)
        self.max_length = max_length
        self.verbose = verbose
        self.grad_clip = grad_clip
        self.check_gradient_conflict = check_gradient_conflict

        if task_weights is None:
            self.task_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        else:
            self.task_weights = torch.tensor(task_weights, dtype=torch.float32, device=device)

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

        self.loss_fns = []
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.loss_fns.append(nn.MSELoss())
            elif cfg['type'] == 'per_residue_classification':
                self.loss_fns.append(nn.CrossEntropyLoss(ignore_index=-100))
            else:
                self.loss_fns.append(nn.CrossEntropyLoss())

        self.history = {
            "train_loss": [], 
            "val_metrics": [], 
            "test_metrics": [],
            "per_task_losses": [],
            "per_task_grad_norms": [],
            "gradient_conflicts": []
        }
        self.best = {"scores": [None]*self.num_tasks, "epochs": [None]*self.num_tasks}
        self.save_dir = ensure_dir(save_dir) if save_dir else None
        
        if self.verbose:
            loader_sizes = [len(loader) for loader in self.train_loaders]
            print(f"[Engine] Train loader sizes (batches): {loader_sizes}")
            print(f"[Engine] Task weights: {[f'{w:.2f}' for w in self.task_weights.cpu().numpy()]}")
            print(f"[Engine] Task names: {[cfg['name'] for cfg in task_configs]}")
            print(f"[Engine] Gradient conflict detection: {self.check_gradient_conflict}")

    def forward_task(self, embeddings, task_idx):
        return self.task_heads[task_idx](embeddings)

    def compute_task_loss(self, logits, targets, task_idx):
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
            
        else:
            targets_flat = targets.view(-1)
            return self.loss_fns[task_idx](logits, targets_flat.long())

    def compute_weighted_loss(self, task_idx, logits, targets):
        cfg = self.task_configs[task_idx]
        task_loss = self.compute_task_loss(logits, targets, task_idx)
        
        # Safety check for NaN
        if torch.isnan(task_loss):
            if self.verbose:
                print(f"WARNING: NaN loss detected in Task {task_idx}")
            return torch.tensor(0.0, device=self.device), 0.0, 0.0
        
        if cfg['type'] == 'per_residue_classification':
            num_valid_tokens = (targets != -100).sum().item()
            if num_valid_tokens == 0:
                if self.verbose:
                    print(f"WARNING: No valid tokens in Task {task_idx}")
                normalized_loss = task_loss
            else:
                normalized_loss = task_loss / num_valid_tokens
        else:
            normalized_loss = task_loss
        
        weighted_loss = self.task_weights[task_idx] * normalized_loss
        return weighted_loss, task_loss.detach().item(), normalized_loss.detach().item()

    def analyze_gradient_conflict_between_tasks(self, task_idx_1, task_idx_2):
        """
        Compute cosine similarity between gradients of two tasks
        Returns value in [-1, 1]: positive=aligned, negative=conflicting
        """
        # Get gradient vectors
        grads = []
        for idx in [task_idx_1, task_idx_2]:
            grad_vec = extract_gradient_vector(self.backbone)
            grads.append(grad_vec)
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(grads[0].unsqueeze(0), grads[1].unsqueeze(0)).item()
        return cosine_sim

    def train_one_epoch(self, optimizer):
        self.backbone.train()
        for head in self.task_heads:
            head.train()

        loaders_cycle = [cycle(loader) for loader in self.train_loaders]
        max_batches = max(len(loader) for loader in self.train_loaders)
        
        total_weighted_loss = 0.0
        per_task_stats = {
            'weighted_loss': [0.0] * self.num_tasks,
            'raw_loss': [0.0] * self.num_tasks,
            'normalized_loss': [0.0] * self.num_tasks,
            'grad_norm': [0.0] * self.num_tasks,
            'updates': [0] * self.num_tasks
        }
        
        # Track gradient conflicts
        conflict_matrix = np.zeros((self.num_tasks, self.num_tasks))
        conflict_count = 0
        
        if self.verbose:
            loader_sizes = [len(loader) for loader in self.train_loaders]
            print(f"\n[Epoch] Cyclic training with max_batches={max_batches}")
            print(f"        Loader sizes: {loader_sizes}")

        total_updates = 0
        
        for batch_idx in range(max_batches):
            for task_idx in range(self.num_tasks):
                batch = next(loaders_cycle[task_idx])
                
                input_ids = batch['sequence'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets']['target'].to(self.device)

                optimizer.zero_grad()
                
                per_residue = (self.task_configs[task_idx]['type'] == 'per_residue_classification')
                embeddings = self.backbone(input_ids, attention_mask, per_residue=per_residue)
                logits = self.forward_task(embeddings, task_idx)

                weighted_loss, raw_loss, normalized_loss = self.compute_weighted_loss(task_idx, logits, targets)
                
                # Check for NaN before backward
                if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                    if self.verbose:
                        print(f"WARNING: NaN/Inf loss at batch {batch_idx}, task {task_idx}")
                    continue
                
                weighted_loss.backward()

                # GRADIENT CONFLICT DETECTION
                if self.check_gradient_conflict and batch_idx == 0 and total_updates < DEBUG_INTERVAL:
                    for other_task_idx in range(task_idx):
                        # Compare with previous tasks' gradients
                        cosine_sim = self.analyze_gradient_conflict_between_tasks(task_idx, other_task_idx)
                        conflict_matrix[task_idx, other_task_idx] = cosine_sim
                        conflict_matrix[other_task_idx, task_idx] = cosine_sim
                        
                        if cosine_sim < -0.1:
                            conflict_count += 1

                torch.nn.utils.clip_grad_norm_(
                    list(self.backbone.parameters()) + list(self.task_heads.parameters()),
                    max_norm=self.grad_clip
                )
                
                grad_norm, _, _ = debug_grad_norm(self.backbone)
                
                optimizer.step()

                total_weighted_loss += weighted_loss.item()
                per_task_stats['weighted_loss'][task_idx] += weighted_loss.item()
                per_task_stats['raw_loss'][task_idx] += raw_loss
                per_task_stats['normalized_loss'][task_idx] += normalized_loss
                per_task_stats['grad_norm'][task_idx] += grad_norm
                per_task_stats['updates'][task_idx] += 1
                total_updates += 1

                if total_updates % DEBUG_INTERVAL == 0 and self.verbose:
                    print(f"[Update {total_updates}] Task {task_idx} ({self.task_configs[task_idx]['name']}):")
                    print(f"  Raw: {raw_loss:.4f} | Normalized: {normalized_loss:.4f} | Weighted: {weighted_loss.item():.4f}")
                    print(f"  Grad norm: {grad_norm:.4e}")

        avg_weighted_loss = total_weighted_loss / max(1, total_updates)
        avg_per_task_stats = {}
        for key in ['weighted_loss', 'raw_loss', 'normalized_loss', 'grad_norm']:
            avg_per_task_stats[key] = [
                per_task_stats[key][i] / max(1, per_task_stats['updates'][i]) 
                for i in range(self.num_tasks)
            ]
        
        self.history["per_task_losses"].append({
            "raw": avg_per_task_stats['raw_loss'],
            "normalized": avg_per_task_stats['normalized_loss'],
            "weighted": avg_per_task_stats['weighted_loss'],
            "updates": per_task_stats['updates']
        })
        self.history["per_task_grad_norms"].append(avg_per_task_stats['grad_norm'])
        self.history["gradient_conflicts"].append(conflict_matrix.tolist())
        
        if self.verbose:
            print(f"\n[Epoch Summary]")
            print(f"  Total updates: {total_updates}")
            print(f"  Updates per task: {per_task_stats['updates']}")
            for i in range(self.num_tasks):
                print(f"  Task {i} ({self.task_configs[i]['name']}):")
                print(f"    Raw loss: {avg_per_task_stats['raw_loss'][i]:.4f}")
                print(f"    Normalized loss: {avg_per_task_stats['normalized_loss'][i]:.4f}")
                print(f"    Avg grad norm: {avg_per_task_stats['grad_norm'][i]:.4e}")
            
            # Print gradient conflict matrix
            if self.check_gradient_conflict:
                print(f"\n[Gradient Conflict Matrix] (cosine similarity)")
                print(f"  Positive = aligned | Negative = conflicting")
                for i in range(self.num_tasks):
                    row_str = f"  Task {i}: "
                    for j in range(self.num_tasks):
                        if i == j:
                            row_str += f"[  1.00  ] "
                        else:
                            val = conflict_matrix[i, j]
                            row_str += f"[{val:6.2f}] "
                    print(row_str)
        
        return avg_weighted_loss

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