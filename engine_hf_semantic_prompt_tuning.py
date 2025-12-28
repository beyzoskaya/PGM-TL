import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle
import os
import csv
import random
import numpy as np

from scipy.stats import spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, mean_squared_error

def multitask_collate_fn(batch, tokenizer):
    # --- 1. Sequence Validation ---
    raw_seqs = []
    for idx, item in enumerate(batch):
        seq = item.get('sequence')
        if not isinstance(seq, str):
            raise ValueError(f"[Collate Error] Sample {idx} sequence is not a string! Found: {type(seq)}")
        if len(seq.strip()) == 0:
            raise ValueError(f"[Collate Error] Sample {idx} sequence is empty/whitespace!")
        raw_seqs.append(seq)

    spaced_seqs = [" ".join(list(s)) for s in raw_seqs]
    inputs = tokenizer(spaced_seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    
    # --- 2. Target Processing ---
    raw_targets = [item['targets']['target'] for item in batch]
    target_tensor = None
    first_target = raw_targets[0]

    if first_target is None:
        raise ValueError("[Collate Error] Found 'None' in targets. Missing labels are not allowed during training!")

    elif isinstance(first_target, list): 
        # SSP Task
        batch_size = len(raw_targets)
        max_seq_len = inputs['input_ids'].shape[1] 
        target_tensor = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
        for i, t_seq in enumerate(raw_targets):
            valid_len = min(len(t_seq), max_seq_len - 2) 
            if valid_len > 0: target_tensor[i, 1 : 1+valid_len] = torch.tensor(t_seq[:valid_len], dtype=torch.long)
                
    elif isinstance(first_target, float): 
        # Thermostability
        if any(not isinstance(t, float) for t in raw_targets):
             raise ValueError("[Collate Error] Regression batch contains non-float targets!")
        target_tensor = torch.tensor(raw_targets, dtype=torch.float32).unsqueeze(1)
        
    elif isinstance(first_target, int): 
        # Cloning
        if any(not isinstance(t, int) for t in raw_targets):
             raise ValueError("[Collate Error] Classification batch contains non-int targets!")
        target_tensor = torch.tensor(raw_targets, dtype=torch.long)
        
    else: 
        raise ValueError(f"[Collate Error] Unknown target type: {type(first_target)}. Cannot process.")
    
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'targets': target_tensor}

class TaskPromptedEngine(nn.Module):
    def __init__(self, backbone, task_configs, train_sets, valid_sets, 
                 test_sets=None, batch_size=8, device='cuda', save_dir=".",
                 init_strategy="semantic"):
        super().__init__()
        self.backbone = backbone.to(device)
        self.device = device
        self.task_configs = task_configs
        self.save_dir = save_dir
        self.log_vars = nn.Parameter(torch.zeros(len(task_configs), device=device))

        os.makedirs(save_dir, exist_ok=True)
        self.sigma_log_path = os.path.join(save_dir, "training_sigmas.csv")
        if not os.path.exists(self.sigma_log_path):
            with open(self.sigma_log_path, 'w', newline='') as f:
                csv.writer(f).writerow(["Epoch", "Step"] + [t['name'] for t in task_configs])

        hidden_dim = backbone.hidden_size
        
        # --- PROMPTS ---
        print(f"[Engine] Initializing {len(task_configs)} Task Prompts (Strategy: {init_strategy})...")
        self.task_prompts = nn.ParameterList()
        cls_vec = None
        if init_strategy == "semantic":
            cls_vec = backbone.get_cls_embedding(device).detach().unsqueeze(0)

        for _ in task_configs:
            p = nn.Parameter(torch.zeros(1, 1, hidden_dim).to(device))
            if init_strategy == "semantic" and cls_vec is not None:
                p.data.copy_(cls_vec)
                p.data.add_(torch.randn_like(p) * 0.001)
            else:
                nn.init.normal_(p, std=0.02)
            self.task_prompts.append(p)

        # --- HEADS ---
        self.heads = nn.ModuleList()
        self.loss_fns = []
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.heads.append(nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_dim, 1)).to(device))
                self.loss_fns.append(nn.MSELoss())
            elif cfg['type'] == 'token_classification':
                self.heads.append(nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_dim, cfg['num_labels'])).to(device))
                self.loss_fns.append(nn.CrossEntropyLoss(ignore_index=-100))
            else:
                self.heads.append(nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_dim, cfg['num_labels'])).to(device))
                self.loss_fns.append(nn.CrossEntropyLoss())

        tokenizer = backbone.tokenizer
        self.train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in train_sets]
        self.valid_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in valid_sets] if valid_sets else None
        self.test_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in test_sets] if test_sets else None

    def forward(self, input_ids, attention_mask, task_idx, debug=False):
        batch_size = input_ids.shape[0]
        raw_prompt = self.task_prompts[task_idx] 
        task_prompt_embeds = raw_prompt.expand(batch_size, -1, -1)
        
        if debug:
            print(f"  [Engine] Processing Task: {self.task_configs[task_idx]['name']} (ID: {task_idx})")
        
        t_type = 'token' if self.task_configs[task_idx]['type'] == 'token_classification' else 'sequence'

        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            task_prompt_embeds=task_prompt_embeds,
            task_type=t_type,
            debug=debug
        )
        
        if debug:
            print(f"  [Engine] Received Output from Backbone: {outputs.shape}")
        
        return self.heads[task_idx](outputs)

    def _project_conflicting(self, grads):
        pc_grads = [g.clone() for g in grads]
        indices = list(range(len(pc_grads)))
        random.shuffle(indices)
        conflicts = 0
        for i in indices:
            for j in indices:
                if i == j: continue
                g_i = pc_grads[i]; g_j = grads[j]
                dot = torch.dot(g_i, g_j)
                if dot < 0:
                    conflicts += 1
                    g_j_norm = torch.dot(g_j, g_j)
                    if g_j_norm > 1e-8:
                        pc_grads[i] -= (dot / g_j_norm) * g_j
        return torch.stack(pc_grads).sum(dim=0), conflicts

    def train_one_epoch(self, optimizer, scheduler, epoch_index=1):
        self.backbone.train()
        for h in self.heads: h.train()
        
        loader_lens = [len(l) for l in self.train_loaders]
        max_steps = max(loader_lens)
        loss_scales = [length / max_steps for length in loader_lens]
        
        if epoch_index == 1:
            print(f"\n[Balance] Dataset Repetition Compensation:")
            for i, cfg in enumerate(self.task_configs):
                repeats = max_steps / loader_lens[i]
                print(f"  - {cfg['name']:<15}: {loader_lens[i]} batches. Repeats {repeats:.2f}x. Loss Scale: {loss_scales[i]:.4f}")

        iterators = [cycle(l) for l in self.train_loaders]
        
        epoch_loss = 0
        print(f"\n[Train] Prompt-PCGrad Epoch {epoch_index} starting...")
        
        for step in range(max_steps):
            task_grads = []
            step_loss_total = 0
            
            debug_stats = []
            should_print = (step == 0) or (step % 500 == 0)

            for i in range(len(self.task_configs)):
                optimizer.zero_grad() 
                batch = next(iterators[i])
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                debug_backbone = (step == 0 and i == 0 and epoch_index == 1)
                logits = self.forward(input_ids, mask, task_idx=i, debug=debug_backbone)
                
                is_token = (self.task_configs[i]['type'] == 'token_classification')
                if is_token: 
                    loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                else: 
                    loss = self.loss_fns[i](logits, targets)
                
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = (precision * loss) + self.log_vars[i]
                
                balanced_loss = weighted_loss * loss_scales[i]
                
                if should_print:
                    debug_stats.append({
                        'name': self.task_configs[i]['name'],
                        'raw': loss.item(),
                        'uncert': weighted_loss.item(),
                        'scale': loss_scales[i],
                        'final': balanced_loss.item()
                    })

                balanced_loss.backward()
                step_loss_total += balanced_loss.item()
                
                grads = []
                for p in self.parameters():
                    if p.requires_grad:
                        if p.grad is not None: grads.append(p.grad.detach().flatten())
                        else: grads.append(torch.zeros(p.numel(), device=self.device))
                task_grads.append(torch.cat(grads))
            
            if should_print:
                print(f"\n[Debug Step {step}] Loss Breakdown:")
                print(f"  {'Task':<15} | {'Raw Loss':<10} | {'Uncert Loss':<12} | {'Freq Scale':<10} | {'Final Loss':<10}")
                print("-" * 70)
                for s in debug_stats:
                    print(f"  {s['name']:<15} | {s['raw']:<10.4f} | {s['uncert']:<12.4f} | {s['scale']:<10.4f} | {s['final']:<10.4f}")
                print("-" * 70)

            final_grad, conflicts = self._project_conflicting(task_grads)
            
            optimizer.zero_grad()
            idx = 0
            for p in self.parameters():
                if p.requires_grad:
                    numel = p.numel()
                    p.grad = final_grad[idx : idx+numel].view_as(p)
                    idx += numel
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            
            epoch_loss += step_loss_total
            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0] if scheduler else 0
                print(f"  Step {step}/{max_steps} | Loss: {step_loss_total:.4f} | Conflicts: {conflicts} | LR: {lr:.2e}")

        return {"avg_loss": epoch_loss / max_steps}

    def evaluate(self, loader_list=None, split_name="Validation"):
        if loader_list is None: loader_list = self.valid_loaders
        self.backbone.eval()
        for h in self.heads: h.eval()
        
        # We will return a dictionary of dictionaries for better logging
        # e.g., {'Thermo': {'MSE': 0.2, 'Spearman': 0.8}, ...}
        results_log = {}
        
        # Raw data to return for plotting
        raw_plot_data = {
            'Thermo': {'true': [], 'pred': []},
            'SSP': {'true': [], 'pred': []},
            'Cloning': {'true': [], 'pred': [], 'probs': []}
        }
        
        print(f"\n[{split_name}] Evaluating & Collecting Data...")
        
        with torch.no_grad():
            for i, loader in enumerate(loader_list):
                name = self.task_configs[i]['name']
                is_token = (self.task_configs[i]['type'] == 'token_classification')
                
                all_preds = []
                all_targets = []
                all_probs = [] # For AUC/Cloning
                
                total_loss=0; total=0
                
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    logits = self.forward(input_ids, mask, task_idx=i)
                    
                    if is_token: # SSP
                        # Flatten for metrics
                        p = logits.argmax(dim=-1).view(-1).cpu().numpy()
                        t = targets.view(-1).cpu().numpy()
                        
                        # Filter out -100
                        mask_np = t != -100
                        all_preds.extend(p[mask_np])
                        all_targets.extend(t[mask_np])
                        
                        # Data for plotting (Batch Level)
                        p_batch = logits.argmax(dim=-1).cpu().numpy()
                        t_batch = targets.cpu().numpy()
                        for b in range(t_batch.shape[0]):
                            valid = t_batch[b] != -100
                            raw_plot_data['SSP']['true'].extend(t_batch[b][valid])
                            raw_plot_data['SSP']['pred'].extend(p_batch[b][valid])
                            
                    else:
                        loss = self.loss_fns[i](logits, targets)
                        total_loss += loss.item() * input_ids.size(0)
                        total += input_ids.size(0)

                        if self.task_configs[i]['type'] == 'regression': # Thermo
                            p = logits.view(-1).cpu().numpy()
                            t = targets.view(-1).cpu().numpy()
                            all_preds.extend(p)
                            all_targets.extend(t)
                            
                            raw_plot_data['Thermo']['true'].extend(t)
                            raw_plot_data['Thermo']['pred'].extend(p)
                            
                        else: # Cloning
                            p = logits.argmax(dim=1).cpu().numpy()
                            t = targets.cpu().numpy()
                            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                            
                            all_preds.extend(p)
                            all_targets.extend(t)
                            all_probs.extend(probs)
                            
                            raw_plot_data['Cloning']['true'].extend(t)
                            raw_plot_data['Cloning']['pred'].extend(p)
                            raw_plot_data['Cloning']['probs'].extend(probs)

                # --- CALCULATE SCIENTIFIC METRICS ---
                metrics = {}
                
                if self.task_configs[i]['type'] == 'regression':
                    mse = mean_squared_error(all_targets, all_preds)
                    try:
                        spearman = spearmanr(all_targets, all_preds).correlation
                    except: spearman = 0.0
                    
                    metrics['MSE'] = mse
                    metrics['Spearman'] = spearman
                    print(f"  {name}: MSE={mse:.4f} | Spearman={spearman:.4f}")
                    
                elif self.task_configs[i]['type'] == 'token_classification':
                    acc = accuracy_score(all_targets, all_preds)
                    f1 = f1_score(all_targets, all_preds, average='macro')
                    
                    metrics['Acc'] = acc
                    metrics['F1'] = f1
                    print(f"  {name}: Acc={acc:.4f} | F1={f1:.4f}")
                    
                else: # Cloning
                    acc = accuracy_score(all_targets, all_preds)
                    mcc = matthews_corrcoef(all_targets, all_preds)
                    
                    metrics['Acc'] = acc
                    metrics['MCC'] = mcc
                    print(f"  {name}: Acc={acc:.4f} | MCC={mcc:.4f}")
                
                results_log[name] = metrics
                
        return results_log, raw_plot_data