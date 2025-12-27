import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import cycle
import os
import csv
import numpy as np

def multitask_collate_fn(batch, tokenizer):
    sequences = [item['sequence'] if isinstance(item['sequence'], str) else " " for item in batch]
    # Add spaces for ProtBert tokenization
    sequences = [" ".join(list(s)) for s in sequences]
    
    inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    
    raw_targets = [item['targets']['target'] for item in batch]
    target_tensor = None
    
    # Logic for different target types (List=SSP, Float=Thermo, Int=Cloning)
    if raw_targets[0] is not None and isinstance(raw_targets[0], list): # SSP
        batch_size = len(raw_targets)
        max_seq_len = inputs['input_ids'].shape[1] 
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

class CascadeMultiTaskEngine(nn.Module):
    def __init__(self, backbone, task_configs, train_sets, valid_sets, 
                 test_sets=None, batch_size=8, device='cuda', save_dir="."):
        super().__init__()
        self.backbone = backbone.to(device)
        self.device = device
        self.task_configs = task_configs
        self.save_dir = save_dir
        
        # Uncertainty Weighting Parameters (One for each task)
        self.log_vars = nn.Parameter(torch.zeros(len(task_configs), device=device))

        os.makedirs(save_dir, exist_ok=True)
        self.sigma_log_path = os.path.join(save_dir, "training_dynamics_sigmas.csv")
        if not os.path.exists(self.sigma_log_path):
            with open(self.sigma_log_path, 'w', newline='') as f:
                csv.writer(f).writerow(["Epoch", "Step"] + [f"Sigma_{cfg['name']}" for cfg in task_configs])

        hidden_dim = backbone.hidden_size # 1024 for ProtBert
        
        # --- 1. THE STRUCTURAL EXPERT (Source of the Cascade) ---
        # We enforce that Index 1 is Secondary Structure
        self.ssp_dim = task_configs[1]['num_labels'] 
        self.head_ssp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.ssp_dim)
        ).to(device)

        # --- 2. THE INTERACTION ADAPTER (The Novelty) ---
        # Projects (Batch, Len, 8) -> (Batch, Len, 1024)
        # Allows structure info to mix with sequence embeddings
        self.cascade_adapter = nn.Sequential(
            nn.Linear(self.ssp_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim) # Crucial for stable addition
        ).to(device)

        # --- 3. THE FUNCTIONAL EXPERTS (Receivers) ---
        # Thermostability (Regression) - Index 0
        self.head_thermo = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
        ).to(device)
        
        # Cloning (Binary Classification) - Index 2
        self.head_cloning = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(hidden_dim, 2)
        ).to(device)

        # Loss Functions
        self.loss_fns = [
            nn.MSELoss(),                           # 0: Thermo
            nn.CrossEntropyLoss(ignore_index=-100), # 1: SSP
            nn.CrossEntropyLoss()                   # 2: Cloning
        ]

        # Loaders
        tokenizer = backbone.tokenizer
        self.train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in train_sets]
        self.valid_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in valid_sets] if valid_sets else None
        self.test_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in test_sets] if test_sets else None

    def forward(self, input_ids, attention_mask, debug=False):
        """
        The Directed Acyclic Graph (DAG) Forward Pass.
        Sequence -> Structure -> Function
        """
        # A. Base Embeddings from ProtBert
        # [Batch, SeqLen, 1024]
        base_emb = self.backbone(input_ids, attention_mask, task_type='token')
        
        # B. Structure Prediction (The Guide)
        # [Batch, SeqLen, 8]
        ssp_logits = self.head_ssp(base_emb)
        
        if debug:
            print(f"    [Cascade Debug] Base Emb: {base_emb.shape} | SSP Logits: {ssp_logits.shape}")

        # C. The Cascade Injection (Novelty)
        # 1. Project 8 -> 1024
        ssp_features = self.cascade_adapter(ssp_logits)
        
        # 2. Add back to Base Embeddings (Residual Connection)
        # enriched_emb represents "Sequence + Structure"
        enriched_emb = base_emb + ssp_features
        
        if debug:
            print(f"    [Cascade Debug] Enriched Emb (After Injection): {enriched_emb.shape}")

        # D. Pooling (Manually implemented for Enriched Embeddings)
        # We must pool the *enriched* embeddings to get sequence-level vectors
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(enriched_emb.size()).float()
        sum_embeddings = torch.sum(enriched_emb * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_emb = sum_embeddings / sum_mask
        
        # E. Functional Predictions
        thermo_pred = self.head_thermo(pooled_emb)
        cloning_pred = self.head_cloning(pooled_emb)

        return ssp_logits, thermo_pred, cloning_pred

    def log_sigmas(self, epoch, step):
        sigmas = torch.exp(self.log_vars).detach().cpu().numpy().tolist()
        with open(self.sigma_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, step] + sigmas)

    def train_one_epoch(self, optimizer, scheduler=None, epoch_index=1):
        self.backbone.train()
        self.head_ssp.train(); self.cascade_adapter.train()
        self.head_thermo.train(); self.head_cloning.train()

        loader_lens = [len(l) for l in self.train_loaders]
        max_steps = max(loader_lens)
        iterators = [cycle(l) for l in self.train_loaders]
        
        epoch_loss = 0
        
        print(f"\n[Train] Cascade Epoch {epoch_index} starting (Steps: {max_steps})...")
        
        for step in range(max_steps):
            optimizer.zero_grad()
            step_loss = 0
            
            # Iterate through all 3 tasks
            for i in range(len(self.task_configs)):
                batch = next(iterators[i])
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Debug only on first step of first epoch
                debug_now = (step == 0 and i == 0 and epoch_index == 1)
                
                # Run the full Cascade
                ssp_pred, thermo_pred, clone_pred = self.forward(input_ids, mask, debug=debug_now)
                
                # Select Loss based on which Task Index 'i' we are processing
                if i == 0: # Thermostability
                    loss = self.loss_fns[0](thermo_pred, targets)
                elif i == 1: # SSP
                    loss = self.loss_fns[1](ssp_pred.view(-1, self.ssp_dim), targets.view(-1))
                elif i == 2: # Cloning
                    loss = self.loss_fns[2](clone_pred, targets)
                
                # Uncertainty Weighting
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = (precision * loss) + self.log_vars[i]
                
                weighted_loss.backward()
                step_loss += weighted_loss.item()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            
            # Step Scheduler if provided
            if scheduler is not None:
                scheduler.step()
            
            epoch_loss += step_loss
            
            # Logging
            if step % 50 == 0:
                lr_curr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                print(f"  Step {step}/{max_steps} | Loss: {step_loss:.4f} | LR: {lr_curr:.2e}")
                
            if step % 100 == 0:
                self.log_sigmas(epoch_index, step)

        return {"avg_loss": epoch_loss / max_steps}

    def evaluate(self, loader_list=None, split_name="Validation"):
        if loader_list is None: loader_list = self.valid_loaders
        self.backbone.eval()
        self.head_ssp.eval(); self.head_thermo.eval(); self.head_cloning.eval()
        
        results = {}
        print(f"\n[{split_name}] Evaluating...")
        
        with torch.no_grad():
            for i, loader in enumerate(loader_list):
                total_loss = 0; total = 0; correct = 0
                name = self.task_configs[i]['name']
                
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    ssp_p, thermo_p, clone_p = self.forward(input_ids, mask)
                    
                    if i == 0: # Thermo
                        loss = self.loss_fns[0](thermo_p, targets)
                        total_loss += loss.item() * input_ids.size(0); total += input_ids.size(0)
                    elif i == 1: # SSP
                        loss = self.loss_fns[1](ssp_p.view(-1, self.ssp_dim), targets.view(-1))
                        total_loss += loss.item() * input_ids.size(0); total += input_ids.size(0)
                        # Acc
                        p = ssp_p.argmax(dim=-1).view(-1); t = targets.view(-1)
                        mask_valid = t != -100
                        if mask_valid.sum() > 0:
                            correct += (p[mask_valid] == t[mask_valid]).sum().item()
                            total += mask_valid.sum().item()
                    elif i == 2: # Cloning
                        loss = self.loss_fns[2](clone_p, targets)
                        total_loss += loss.item() * input_ids.size(0); total += input_ids.size(0)
                        correct += (clone_p.argmax(dim=1) == targets).sum().item()

                if i == 0: results[name] = f"MSE: {total_loss/total:.4f}"
                else: results[name] = f"Acc: {correct/(total if total>0 else 1):.4f}"
                print(f"  {name}: {results[name]}")
        return results