import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from itertools import cycle
import os
import csv
import datetime
import gc 

def multitask_collate_fn(batch, tokenizer):
    sequences = []
    for item in batch:
        seq = item['sequence']
        if isinstance(seq, str):
            sequences.append(" ".join(list(seq)))
        else:
            sequences.append("")
    
    inputs = tokenizer(
        sequences, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=1024 
    )
    
    raw_targets = [item['targets']['target'] for item in batch]
    target_tensor = None
    
    if raw_targets[0] is not None and isinstance(raw_targets[0], list):
        batch_size = len(raw_targets)
        max_seq_len = inputs['input_ids'].shape[1] 
        target_tensor = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
        for i, t_seq in enumerate(raw_targets):
            valid_len = min(len(t_seq), max_seq_len - 2) 
            if valid_len > 0:
                target_tensor[i, 1 : 1+valid_len] = torch.tensor(t_seq[:valid_len], dtype=torch.long)

    elif raw_targets[0] is not None and isinstance(raw_targets[0], float):
        target_tensor = torch.tensor(raw_targets, dtype=torch.float32).unsqueeze(1)

    elif raw_targets[0] is not None and isinstance(raw_targets[0], int):
        target_tensor = torch.tensor(raw_targets, dtype=torch.long)
    else:
        target_tensor = torch.zeros(len(raw_targets))

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'targets': target_tensor
    }

class MultiTaskEngineUncertanityWeighting(nn.Module):
    def __init__(self, backbone, task_configs, train_sets, valid_sets, 
                 test_sets=None, batch_size=8, device='cuda', save_dir="."):
        super().__init__()
        self.backbone = backbone.to(device)
        self.device = device
        self.task_configs = task_configs
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.log_vars = nn.Parameter(torch.zeros(len(task_configs), device=device))

        os.makedirs(save_dir, exist_ok=True)
        
        # 1. SIGMA LOGGING (Smart Append)
        self.sigma_log_path = os.path.join(save_dir, "training_dynamics_sigmas.csv")
        # Only write header if file doesn't exist
        if not os.path.exists(self.sigma_log_path):
            with open(self.sigma_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["Epoch", "Step"] + [f"Sigma_{cfg['name']}" for cfg in task_configs] + [f"Loss_{cfg['name']}" for cfg in task_configs]
                writer.writerow(header)

        # 2. GRADIENT LOGGING (Smart Append)
        self.grad_log_path = os.path.join(save_dir, "gradient_conflict_log.txt")
        # Just append a "Resumed" message
        with open(self.grad_log_path, 'a') as f:
            f.write(f"\n--- SESSION RESUMED: {datetime.datetime.now()} ---\n")

        # Heads & Losses
        self.heads = nn.ModuleList()
        self.loss_fns = []
        hidden_dim = backbone.hidden_size
        
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.heads.append(nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.Tanh(),
                    nn.Linear(hidden_dim // 2, 1)
                ).to(device))
                self.loss_fns.append(nn.MSELoss())
            elif cfg['type'] in ['token_classification', 'per_residue_classification']:
                self.heads.append(nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, cfg['num_labels'])
                ).to(device))
                self.loss_fns.append(nn.CrossEntropyLoss(ignore_index=-100))
            else: 
                self.heads.append(nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, cfg['num_labels'])
                ).to(device))
                self.loss_fns.append(nn.CrossEntropyLoss())

        tokenizer = backbone.tokenizer
        self.train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in train_sets]
        self.valid_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in valid_sets] if valid_sets else None
        self.test_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in test_sets] if test_sets else None

    def analyze_gradients(self, losses, step, epoch):
        grads = []
        trainable_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if not trainable_params: return

        for loss in losses:
            g = torch.autograd.grad(loss, trainable_params, retain_graph=True, allow_unused=True)
            flat_g = torch.cat([x.flatten() for x in g if x is not None])
            grads.append(flat_g)
        
        log_msg = f"\n[Epoch {epoch} Step {step}] Gradient Cosine Similarity:\n"
        num_tasks = len(grads)
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                sim = F.cosine_similarity(grads[i].unsqueeze(0), grads[j].unsqueeze(0))
                t1 = self.task_configs[i]['name']
                t2 = self.task_configs[j]['name']
                log_msg += f"  > {t1} vs {t2}: {sim.item():.4f}\n"
        
        with open(self.grad_log_path, 'a') as f:
            f.write(log_msg)
        print(log_msg)

    def log_dynamics(self, epoch, step, losses):
        sigmas = torch.exp(self.log_vars).detach().cpu().numpy().tolist()
        raw_losses = [l.item() if isinstance(l, torch.Tensor) else l for l in losses]
        row = [epoch, step] + sigmas + raw_losses
        with open(self.sigma_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def train_one_epoch(self, optimizer, epoch_index=1):
        self.backbone.train()
        for h in self.heads: h.train()

        loader_lens = [len(l) for l in self.train_loaders]
        max_steps = max(loader_lens)
        iterators = [cycle(l) for l in self.train_loaders]
        
        epoch_loss = 0
        task_stats = {i: {'loss': 0.0, 'total': 0, 'correct': 0} for i in range(len(self.task_configs))}
        
        print(f"\n[Train] Starting Epoch {epoch_index} with {max_steps} steps...")
        
        for step in range(max_steps):
            optimizer.zero_grad()
            step_loss_sum = 0
            
            # Reduce analysis frequency slightly to be safe
            should_analyze_grads = (step == 50) or (step % 1000 == 0 and step > 0)
            should_log_sigmas = (step % 100 == 0)

            # --- PATH A: Heavy Analysis (Memory Intensive) ---
            if should_analyze_grads:
                raw_losses_list = []
                for i in range(len(self.task_configs)):
                    batch = next(iterators[i])
                    input_ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    is_token_task = (self.task_configs[i]['type'] in ['token_classification', 'per_residue_classification'])
                    embeddings = self.backbone(input_ids, mask, task_type='token' if is_token_task else 'sequence')
                    logits = self.heads[i](embeddings)
                    
                    if is_token_task: loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                    else: loss = self.loss_fns[i](logits, targets)
                    
                    raw_losses_list.append(loss)
                    
                    # Weighted Backward with Retain Graph
                    precision = torch.exp(-self.log_vars[i])
                    weighted_loss = (precision * loss) + self.log_vars[i]
                    weighted_loss.backward(retain_graph=True)
                    step_loss_sum += weighted_loss.item()
                    
                    # Minimal Stats
                    with torch.no_grad():
                         task_stats[i]['loss'] += loss.item() * input_ids.size(0)
                         task_stats[i]['total'] += input_ids.size(0)

                self.analyze_gradients(raw_losses_list, step, epoch_index)
                if should_log_sigmas: self.log_dynamics(epoch_index, step, raw_losses_list)
                
                # FORCE CLEANUP
                del raw_losses_list
                torch.cuda.empty_cache() 
                gc.collect()

            # --- PATH B: Fast Training (Memory Efficient) ---
            else:
                total_tensor_loss = 0
                temp_raw_losses = []
                
                for i in range(len(self.task_configs)):
                    batch = next(iterators[i])
                    input_ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    is_token_task = (self.task_configs[i]['type'] in ['token_classification', 'per_residue_classification'])
                    
                    embeddings = self.backbone(input_ids, mask, task_type='token' if is_token_task else 'sequence')
                    logits = self.heads[i](embeddings)
                    
                    if is_token_task: loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                    else: loss = self.loss_fns[i](logits, targets)
                    
                    temp_raw_losses.append(loss.item())

                    precision = torch.exp(-self.log_vars[i])
                    weighted_loss = (precision * loss) + self.log_vars[i]
                    
                    # SUM FIRST, THEN BACKWARD ONCE
                    total_tensor_loss += weighted_loss
                    step_loss_sum += weighted_loss.item()
                    
                    with torch.no_grad():
                        bs = input_ids.size(0)
                        task_stats[i]['loss'] += loss.item() * bs
                        if self.task_configs[i]['type'] == 'regression': task_stats[i]['total'] += bs
                        elif is_token_task:
                            preds = logits.argmax(dim=-1).view(-1); lbls = targets.view(-1); mask_valid = lbls != -100
                            if mask_valid.sum() > 0:
                                task_stats[i]['correct'] += (preds[mask_valid] == lbls[mask_valid]).sum().item()
                                task_stats[i]['total'] += mask_valid.sum().item()
                        else:
                            preds = logits.argmax(dim=1)
                            task_stats[i]['correct'] += (preds == targets).sum().item(); task_stats[i]['total'] += bs

                total_tensor_loss.backward() # No retain_graph needed
                if should_log_sigmas: self.log_dynamics(epoch_index, step, temp_raw_losses)

            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
            optimizer.step()
            epoch_loss += step_loss_sum

            if step % 50 == 0:
                sigmas_str = [f"{np.exp(v):.2f}" for v in self.log_vars.detach().cpu().numpy()]
                print(f"  Step {step}/{max_steps} | Combined Loss: {step_loss_sum:.4f} | Sigmas: {sigmas_str}")

        results = {"combined_loss": epoch_loss / max_steps}
        for i, cfg in enumerate(self.task_configs):
            name = cfg['name']
            avg_loss = task_stats[i]['loss'] / max(task_stats[i]['total'], 1)
            if cfg['type'] == 'regression': results[name] = f"MSE: {avg_loss:.4f}"
            else:
                acc = task_stats[i]['correct'] / (task_stats[i]['total'] if task_stats[i]['total'] > 0 else 1)
                results[name] = f"Loss: {avg_loss:.4f} | Acc: {acc:.4f}"
        return results