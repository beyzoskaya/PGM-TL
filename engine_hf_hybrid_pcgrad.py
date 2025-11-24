import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from itertools import cycle
import os
import csv
import random

def multitask_collate_fn(batch, tokenizer):
    sequences = []
    for item in batch:
        seq = item['sequence']
        if isinstance(seq, str): sequences.append(" ".join(list(seq)))
        else: sequences.append("")
    inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    raw_targets = [item['targets']['target'] for item in batch]
    target_tensor = None
    if raw_targets[0] is not None and isinstance(raw_targets[0], list):
        batch_size = len(raw_targets)
        max_seq_len = inputs['input_ids'].shape[1] 
        target_tensor = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
        for i, t_seq in enumerate(raw_targets):
            valid_len = min(len(t_seq), max_seq_len - 2) 
            if valid_len > 0: target_tensor[i, 1 : 1+valid_len] = torch.tensor(t_seq[:valid_len], dtype=torch.long)
    elif raw_targets[0] is not None and isinstance(raw_targets[0], float):
        target_tensor = torch.tensor(raw_targets, dtype=torch.float32).unsqueeze(1)
    elif raw_targets[0] is not None and isinstance(raw_targets[0], int):
        target_tensor = torch.tensor(raw_targets, dtype=torch.long)
    else: target_tensor = torch.zeros(len(raw_targets))
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'targets': target_tensor}

class MultiTaskEngineHybrid(nn.Module):
    def __init__(self, backbone, task_configs, train_sets, valid_sets, 
                 test_sets=None, batch_size=8, device='cuda', save_dir="."):
        super().__init__()
        self.backbone = backbone.to(device)
        self.device = device
        self.task_configs = task_configs
        self.batch_size = batch_size
        self.save_dir = save_dir

        # Uncertainty Parameters
        self.log_vars = nn.Parameter(torch.zeros(len(task_configs), device=device))

        os.makedirs(save_dir, exist_ok=True)
        
        self.sigma_log_path = os.path.join(save_dir, "training_dynamics_sigmas.csv")
        if not os.path.exists(self.sigma_log_path):
            with open(self.sigma_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["Epoch", "Step"] + [f"Sigma_{cfg['name']}" for cfg in task_configs]
                writer.writerow(header)

        # PCGrad Stats
        self.pcgrad_log_path = os.path.join(save_dir, "pcgrad_statistics.csv")
        if not os.path.exists(self.pcgrad_log_path):
            with open(self.pcgrad_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Conflict_Count: How many pairs had negative cosine
                # Magnitude_Red: How much gradient magnitude was removed by projection
                writer.writerow(["Epoch", "Step", "Conflict_Count", "Avg_Conflict_Magnitude"])

        self.heads = nn.ModuleList()
        self.loss_fns = []
        hidden_dim = backbone.hidden_size
        
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.heads.append(nn.Sequential(
                    nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
                ).to(device))
                self.loss_fns.append(nn.MSELoss())
            elif cfg['type'] in ['token_classification', 'per_residue_classification']:
                self.heads.append(nn.Sequential(
                    nn.Dropout(0.1), nn.Linear(hidden_dim, cfg['num_labels'])
                ).to(device))
                self.loss_fns.append(nn.CrossEntropyLoss(ignore_index=-100))
            else: 
                self.heads.append(nn.Sequential(
                    nn.Dropout(0.1), nn.Linear(hidden_dim, cfg['num_labels'])
                ).to(device))
                self.loss_fns.append(nn.CrossEntropyLoss())

        tokenizer = backbone.tokenizer
        self.train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in train_sets]
        self.valid_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in valid_sets] if valid_sets else None
        self.test_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in test_sets] if test_sets else None

    def _pcgrad_project(self, grads):
        """
        Returns:
            final_grad: The summed, projected gradient
            stats: dict containing 'conflicts' (int) and 'magnitude' (float)
        """
        pc_grads = [g.clone() for g in grads]
        indices = list(range(len(pc_grads)))
        random.shuffle(indices)

        conflict_count = 0
        total_projection_magnitude = 0.0

        for i in indices:
            for j in indices:
                if i == j: continue
                
                g_i = pc_grads[i]
                g_j = grads[j] # Compare against original
                
                dot = torch.dot(g_i, g_j)
                
                if dot < 0:
                    conflict_count += 1
                    g_j_norm_sq = torch.dot(g_j, g_j)
                    if g_j_norm_sq > 0:
                        # Calculate projection vector
                        projection = (dot / g_j_norm_sq) * g_j
                        pc_grads[i] -= projection
                        total_projection_magnitude += projection.norm().item()
        
        final_grad = torch.stack(pc_grads).sum(dim=0)
        
        stats = {
            'conflicts': conflict_count,
            'magnitude': total_projection_magnitude
        }
        return final_grad, stats

    def log_pcgrad_stats(self, epoch, step, stats):
        row = [epoch, step, stats['conflicts'], stats['magnitude']]
        with open(self.pcgrad_log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def log_sigmas(self, epoch, step):
        sigmas = torch.exp(self.log_vars).detach().cpu().numpy().tolist()
        row = [epoch, step] + sigmas
        with open(self.sigma_log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def train_one_epoch(self, optimizer, epoch_index=1):
        self.backbone.train()
        for h in self.heads: h.train()

        loader_lens = [len(l) for l in self.train_loaders]
        max_steps = max(loader_lens)
        iterators = [cycle(l) for l in self.train_loaders]
        
        epoch_loss = 0
        task_stats = {i: {'loss': 0.0, 'total': 0, 'correct': 0} for i in range(len(self.task_configs))}
        
        print(f"\n[Train] Hybrid PCGrad Epoch {epoch_index} starting...")
        
        for step in range(max_steps):
            # 1. GATHER GRADIENTS
            task_gradients_flat = []
            step_combined_loss = 0
            
            for i in range(len(self.task_configs)):
                optimizer.zero_grad()
                
                batch = next(iterators[i])
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                is_token = (self.task_configs[i]['type'] in ['token_classification', 'per_residue_classification'])
                
                emb = self.backbone(input_ids, mask, task_type='token' if is_token else 'sequence')
                logits = self.heads[i](emb)
                
                if is_token: loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                else: loss = self.loss_fns[i](logits, targets)
                
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = (precision * loss) + self.log_vars[i]
                
                weighted_loss.backward() 
                step_combined_loss += weighted_loss.item()
                
                # Flatten ALL parameters (Backbone + Heads + Sigmas)
                grads = []
                for p in self.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().flatten())
                    else:
                        grads.append(torch.zeros(p.numel(), device=self.device))
                task_gradients_flat.append(torch.cat(grads))
                
                with torch.no_grad():
                    bs = input_ids.size(0)
                    task_stats[i]['loss'] += loss.item() * bs
                    if self.task_configs[i]['type'] == 'regression': task_stats[i]['total'] += bs
                    elif is_token:
                        preds = logits.argmax(dim=-1).view(-1); lbls = targets.view(-1); mask = lbls != -100
                        if mask.sum() > 0: task_stats[i]['correct'] += (preds[mask] == lbls[mask]).sum().item(); task_stats[i]['total'] += mask.sum().item()
                    else:
                        preds = logits.argmax(dim=1); task_stats[i]['correct'] += (preds == targets).sum().item(); task_stats[i]['total'] += bs

            # 2. PROJECT
            final_flat_grad, pc_stats = self._pcgrad_project(task_gradients_flat)
            
            # 3. APPLY
            optimizer.zero_grad()
            idx = 0
            for p in self.parameters():
                if p.requires_grad:
                    numel = p.numel()
                    p.grad = final_flat_grad[idx : idx + numel].view_as(p)
                    idx += numel
            
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += step_combined_loss

            if step % 50 == 0:
                sigmas_str = [f"{np.exp(v):.2f}" for v in self.log_vars.detach().cpu().numpy()]
                print(f"  Step {step}/{max_steps} | Loss: {step_combined_loss:.4f} | Sigmas: {sigmas_str} | Conflicts: {pc_stats['conflicts']}")
            
            if step % 100 == 0:
                self.log_sigmas(epoch_index, step)
                self.log_pcgrad_stats(epoch_index, step, pc_stats)

        results = {"combined_loss": epoch_loss / max_steps}
        for i, cfg in enumerate(self.task_configs):
            name = cfg['name']
            avg_loss = task_stats[i]['loss'] / max(task_stats[i]['total'], 1)
            if cfg['type'] == 'regression': results[name] = f"MSE: {avg_loss:.4f}"
            else:
                acc = task_stats[i]['correct'] / (task_stats[i]['total'] if task_stats[i]['total'] > 0 else 1)
                results[name] = f"Loss: {avg_loss:.4f} | Acc: {acc:.4f}"
        return results

    def evaluate(self, loader_list=None, split_name="Validation"):
        if loader_list is None: loader_list = self.valid_loaders
        if not loader_list: return {}
        self.backbone.eval()
        for h in self.heads: h.eval()
        metrics_output = {}
        print(f"\n[{split_name}] Evaluating...")
        with torch.no_grad():
            for i, loader in enumerate(loader_list):
                task_name = self.task_configs[i].get('name', f'Task_{i}')
                is_token = (self.task_configs[i]['type'] in ['token_classification', 'per_residue_classification'])
                is_reg = (self.task_configs[i]['type'] == 'regression')
                total_loss = 0; correct = 0; total_samples = 0; total_tokens = 0
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    embeddings = self.backbone(input_ids, mask, task_type='token' if is_token else 'sequence')
                    logits = self.heads[i](embeddings)
                    if is_token:
                        loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                        total_loss += loss.item() * input_ids.size(0); total_samples += input_ids.size(0)
                        preds = logits.argmax(dim=-1).view(-1); lbls = targets.view(-1); mask_valid = lbls != -100
                        if mask_valid.sum() > 0: correct += (preds[mask_valid] == lbls[mask_valid]).sum().item(); total_tokens += mask_valid.sum().item()
                    elif is_reg:
                        loss = self.loss_fns[i](logits, targets)
                        total_loss += loss.item() * input_ids.size(0); total_samples += input_ids.size(0)
                    else:
                        loss = self.loss_fns[i](logits, targets)
                        total_loss += loss.item() * input_ids.size(0); total_samples += input_ids.size(0)
                        preds = logits.argmax(dim=1); correct += (preds == targets).sum().item()

                avg_loss = total_loss / total_samples if total_samples > 0 else 0
                if is_reg:
                    print(f"  {task_name}: MSE (Loss) = {avg_loss:.4f}")
                    metrics_output[task_name] = {"MSE": avg_loss}
                elif is_token:
                    acc = correct / total_tokens if total_tokens > 0 else 0
                    print(f"  {task_name}: Loss = {avg_loss:.4f} | Accuracy = {acc:.4f}")
                    metrics_output[task_name] = {"Loss": avg_loss, "Accuracy": acc}
                else:
                    acc = correct / total_samples if total_samples > 0 else 0
                    print(f"  {task_name}: Loss = {avg_loss:.4f} | Accuracy = {acc:.4f}")
                    metrics_output[task_name] = {"Loss": avg_loss, "Accuracy": acc}
        return metrics_output