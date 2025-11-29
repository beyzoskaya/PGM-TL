import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle
import os
import csv
import random
import numpy as np

def multitask_collate_fn(batch, tokenizer):
    raw_seqs = [item['sequence'] if isinstance(item['sequence'], str) else " " for item in batch]
    spaced_seqs = [" ".join(list(s)) for s in raw_seqs]
    
    inputs = tokenizer(spaced_seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    
    raw_targets = [item['targets']['target'] for item in batch]
    target_tensor = None
    
    if raw_targets[0] is not None and isinstance(raw_targets[0], list): # SSP
        batch_size = len(raw_targets); max_seq_len = inputs['input_ids'].shape[1] 
        target_tensor = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
        for i, t_seq in enumerate(raw_targets):
            valid_len = min(len(t_seq), max_seq_len - 2) 
            if valid_len > 0: target_tensor[i, 1 : 1+valid_len] = torch.tensor(t_seq[:valid_len], dtype=torch.long)
    elif raw_targets[0] is not None and isinstance(raw_targets[0], float): # Thermo
        target_tensor = torch.tensor(raw_targets, dtype=torch.float32).unsqueeze(1)
    elif raw_targets[0] is not None and isinstance(raw_targets[0], int): # Cloning
        target_tensor = torch.tensor(raw_targets, dtype=torch.long)
    else: 
        target_tensor = torch.zeros(len(raw_targets))
    
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'targets': target_tensor}

class TaskPromptedEngine(nn.Module):
    def __init__(self, backbone, task_configs, train_sets, valid_sets, 
                 test_sets=None, batch_size=8, device='cuda', save_dir="."):
        super().__init__()
        self.backbone = backbone.to(device)
        self.device = device
        self.task_configs = task_configs
        self.save_dir = save_dir
        
        # Uncertainty Weighting (Still useful with PCGrad)
        self.log_vars = nn.Parameter(torch.zeros(len(task_configs), device=device))

        os.makedirs(save_dir, exist_ok=True)
        self.sigma_log_path = os.path.join(save_dir, "training_sigmas.csv")
        if not os.path.exists(self.sigma_log_path):
            with open(self.sigma_log_path, 'w', newline='') as f:
                csv.writer(f).writerow(["Epoch", "Step"] + [t['name'] for t in task_configs])

        hidden_dim = backbone.hidden_size
        
        # --- LEARNABLE TASK PROMPTS ---
        # create 3 distinct vectors. Each is size [1, 1, 1024]
        # These act as "Identity Cards" for the tasks.
        print(f"[Engine] Initializing {len(task_configs)} Task Prompts...")
        self.task_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, hidden_dim).to(device)) 
            for _ in task_configs
        ])
        
        # Initialize with small variance to avoid shock
        for p in self.task_prompts:
            nn.init.normal_(p, std=0.02)

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
        # A. Run Backbone (Get Sequence Embeddings)
        # Output: [Batch, SeqLen, 1024]
        outputs = self.backbone(input_ids, attention_mask, task_type='token') 
        
        if debug:
            print(f"  [Prompt Debug] Backbone Output Shape: {outputs.shape}")

        # B. Get Specific Prompt for this Task
        prompt = self.task_prompts[task_idx] # Shape: [1, 1, 1024]
        
        if debug:
            print(f"  [Prompt Debug] Task {task_idx} Prompt Shape: {prompt.shape}")
            print(f"  [Prompt Debug] Prompt Values (First 5): {prompt[0,0,:5].detach().cpu().numpy()}")

        # C. Inject Prompt (The "Context Switch")
        # Logic: We add the Prompt vector to the embeddings.
        # Since prompt is [1,1,1024], it broadcasts to [Batch, SeqLen, 1024]
        # This shifts the entire embedding space in a task-specific direction.
        
        prompted_embeddings = outputs + prompt
        
        # D. Handle Task Types
        if self.task_configs[task_idx]['type'] == 'token_classification':
            # For Structure: We use the full sequence
            final_emb = prompted_embeddings
        else:
            # For Stability/Cloning: We use the [CLS] token (Index 0)
            # We extract the [CLS] token from the PROMPTED embedding
            final_emb = prompted_embeddings[:, 0, :] 
            
        if debug:
            print(f"  [Prompt Debug] Final Input to Head: {final_emb.shape}")
            
        return self.heads[task_idx](final_emb)

    def _project_conflicting(self, grads):
        """PCGrad Logic: Projects gradients to remove conflict."""
        pc_grads = [g.clone() for g in grads]
        indices = list(range(len(pc_grads)))
        random.shuffle(indices)
        conflicts = 0
        
        for i in indices:
            for j in indices:
                if i == j: continue
                g_i = pc_grads[i]
                g_j = grads[j]
                
                # Dot product checks angle
                dot = torch.dot(g_i, g_j)
                
                if dot < 0: # Conflict!
                    conflicts += 1
                    g_j_norm = torch.dot(g_j, g_j)
                    if g_j_norm > 1e-8:
                        # Remove the conflicting component
                        pc_grads[i] -= (dot / g_j_norm) * g_j
                        
        return torch.stack(pc_grads).sum(dim=0), conflicts

    def train_one_epoch(self, optimizer, scheduler, epoch_index=1):
        self.backbone.train()
        for h in self.heads: h.train()
        
        loader_lens = [len(l) for l in self.train_loaders]
        max_steps = max(loader_lens)
        iterators = [cycle(l) for l in self.train_loaders]
        
        epoch_loss = 0
        print(f"\n[Train] Prompt-PCGrad Epoch {epoch_index} starting...")
        
        for step in range(max_steps):
            task_grads = []
            step_loss_total = 0
            
            # --- PCGRAD STEP ---
            # 1. Compute Gradients for each task INDEPENDENTLY
            # Do NOT call optimizer.step() yet!
            
            for i in range(len(self.task_configs)):
                optimizer.zero_grad() # Clean slate for this task
                
                batch = next(iterators[i])
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Debug only first step of first epoch
                debug_now = (step == 0 and i == 0 and epoch_index == 1)
                
                logits = self.forward(input_ids, mask, task_idx=i, debug=debug_now)
                
                is_token = (self.task_configs[i]['type'] == 'token_classification')
                if is_token: loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                else: loss = self.loss_fns[i](logits, targets)
                
                # Apply Uncertainty Weighting
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = (precision * loss) + self.log_vars[i]
                
                weighted_loss.backward() # Calculate gradients
                step_loss_total += weighted_loss.item()
                
                # Collect Flattened Gradients for this task
                grads = []
                for p in self.parameters():
                    if p.requires_grad:
                        if p.grad is not None: grads.append(p.grad.detach().flatten())
                        else: grads.append(torch.zeros(p.numel(), device=self.device))
                task_grads.append(torch.cat(grads))
            
            # 2. Project Gradients (Resolve Conflict)
            final_grad, conflicts = self._project_conflicting(task_grads)
            
            # 3. Apply Final Gradient
            optimizer.zero_grad()
            idx = 0
            for p in self.parameters():
                if p.requires_grad:
                    numel = p.numel()
                    # Reshape flat gradient back to parameter shape
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
        results = {}
        print(f"\n[{split_name}] Evaluating...")
        
        with torch.no_grad():
            for i, loader in enumerate(loader_list):
                name = self.task_configs[i]['name']
                is_token = (self.task_configs[i]['type'] == 'token_classification')
                total_loss=0; correct=0; total=0
                
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    logits = self.forward(input_ids, mask, task_idx=i)
                    
                    if is_token:
                        loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                        p = logits.argmax(dim=-1).view(-1); t = targets.view(-1); m = t!=-100
                        if m.sum()>0: correct += (p[m]==t[m]).sum().item(); total += m.sum().item()
                    else:
                        loss = self.loss_fns[i](logits, targets)
                        if self.task_configs[i]['type'] == 'regression': total += input_ids.size(0)
                        else: correct += (logits.argmax(dim=1)==targets).sum().item(); total += input_ids.size(0)
                    
                    total_loss += loss.item() * input_ids.size(0)
                
                avg = total_loss / (total if total>0 else 1)
                if self.task_configs[i]['type'] == 'regression': results[name] = f"MSE: {avg:.4f}"
                else: results[name] = f"Acc: {correct/(total if total>0 else 1):.4f}"
                print(f"  {name}: {results[name]}")
        return results