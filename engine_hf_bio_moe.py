import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import cycle
import os
import csv
import numpy as np

# --- BIOPHYSICAL UTILS ---
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

class BioPropertyFeaturizer:
    @staticmethod
    def get_features(sequences, device):
        feats = []
        for seq in sequences:
            s = seq.replace(" ", "")
            if len(s) == 0: 
                feats.append([0, 0, 0])
                continue
            hydropathy = sum(KYTE_DOOLITTLE.get(aa, 0) for aa in s) / len(s)
            pos = s.count('R') + s.count('K')
            neg = s.count('D') + s.count('E')
            charge = (pos - neg) / len(s)
            arom = (s.count('F') + s.count('W') + s.count('Y')) / len(s)
            feats.append([hydropathy, charge, arom])
        return torch.tensor(feats, dtype=torch.float32).to(device)

def multitask_collate_fn(batch, tokenizer):
    raw_seqs = [item['sequence'] if isinstance(item['sequence'], str) else " " for item in batch]
    spaced_seqs = [" ".join(list(s)) for s in raw_seqs]
    inputs = tokenizer(spaced_seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    inputs['raw_sequences'] = raw_seqs 
    
    raw_targets = [item['targets']['target'] for item in batch]
    target_tensor = None
    if raw_targets[0] is not None and isinstance(raw_targets[0], list):
        batch_size = len(raw_targets); max_seq_len = inputs['input_ids'].shape[1] 
        target_tensor = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
        for i, t_seq in enumerate(raw_targets):
            valid_len = min(len(t_seq), max_seq_len - 2) 
            if valid_len > 0: target_tensor[i, 1 : 1+valid_len] = torch.tensor(t_seq[:valid_len], dtype=torch.long)
    elif raw_targets[0] is not None and isinstance(raw_targets[0], float):
        target_tensor = torch.tensor(raw_targets, dtype=torch.float32).unsqueeze(1)
    elif raw_targets[0] is not None and isinstance(raw_targets[0], int):
        target_tensor = torch.tensor(raw_targets, dtype=torch.long)
    else: target_tensor = torch.zeros(len(raw_targets))
    inputs['targets'] = target_tensor
    return inputs

class BottleneckAdapter(nn.Module):
    def __init__(self, input_dim, reduction_factor=4):
        super().__init__()
        bottleneck_dim = input_dim // reduction_factor
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        out = self.down(x)
        out = self.act(out)
        out = self.up(out)
        return self.norm(residual + out)

class BioMoE_Engine(nn.Module):
    def __init__(self, backbone, task_configs, train_sets, valid_sets, 
                 test_sets=None, batch_size=8, device='cuda', save_dir="."):
        super().__init__()
        self.backbone = backbone.to(device)
        self.device = device
        self.task_configs = task_configs
        self.save_dir = save_dir
        self.log_vars = nn.Parameter(torch.zeros(len(task_configs), device=device))

        os.makedirs(save_dir, exist_ok=True)
        self.sigma_log_path = os.path.join(save_dir, "training_dynamics_sigmas.csv")
        if not os.path.exists(self.sigma_log_path):
            with open(self.sigma_log_path, 'w', newline='') as f:
                csv.writer(f).writerow(["Epoch", "Step"] + [f"Sigma_{cfg['name']}" for cfg in task_configs])

        hidden_dim = backbone.hidden_size
        
        # 1. EXPERTS
        self.experts = nn.ModuleList([
            BottleneckAdapter(hidden_dim, reduction_factor=4).to(device) 
            for _ in task_configs
        ])

        # 2. BIO-ROUTER (IMPROVED with Norm)
        self.router_norm = nn.LayerNorm(hidden_dim).to(device)
        self.bio_router = nn.Sequential(
            nn.Linear(hidden_dim + 3, 256),
            nn.Tanh(),
            nn.Linear(256, len(task_configs))
        ).to(device)

        # 3. HEADS
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

    def forward(self, input_ids, attention_mask, raw_sequences, task_idx=None, debug=False):
        # A. Backbone (Frozen)
        with torch.no_grad():
            base_emb = self.backbone(input_ids, attention_mask, task_type='token')
            cls_emb = base_emb[:, 0, :]

        # B. Bio-Features
        bio_feats = BioPropertyFeaturizer.get_features(raw_sequences, self.device)
        
        # C. Bio-Router
        # Normalize CLS so it doesn't overpower bio-feats
        norm_cls = self.router_norm(cls_emb) 
        router_input = torch.cat([norm_cls, bio_feats], dim=1)
        
        routing_logits = self.bio_router(router_input)
        routing_weights = torch.softmax(routing_logits, dim=1)
        
        if debug: print(f"  [Bio-MoE] Routing: {routing_weights[0].detach().cpu().numpy()}")

        # D. Run Experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(base_emb))
        stacked_experts = torch.stack(expert_outputs, dim=0)
        
        # E. Fuse
        weights_permuted = routing_weights.T.unsqueeze(-1).unsqueeze(-1)
        fused_emb = (stacked_experts * weights_permuted).sum(dim=0)

        # F. Prediction Head
        is_token = (self.task_configs[task_idx]['type'] == 'token_classification')
        if is_token:
            final_rep = fused_emb
        else:
            mask_exp = attention_mask.unsqueeze(-1).expand(fused_emb.size()).float()
            final_rep = torch.sum(fused_emb * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
            
        logits = self.heads[task_idx](final_rep)
        return logits, routing_weights

    def log_sigmas(self, epoch, step):
        sigmas = torch.exp(self.log_vars).detach().cpu().numpy().tolist()
        with open(self.sigma_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, step] + sigmas)

    def train_one_epoch(self, optimizer, scheduler, epoch_index=1):
        self.backbone.eval()
        for m in self.experts: m.train()
        for m in self.heads: m.train()
        self.bio_router.train()
        
        loader_lens = [len(l) for l in self.train_loaders]
        max_steps = max(loader_lens)
        iterators = [cycle(l) for l in self.train_loaders]
        
        epoch_loss = 0
        print(f"\n[Train] Bio-MoE Epoch {epoch_index} starting...")
        
        for step in range(max_steps):
            optimizer.zero_grad()
            step_loss = 0
            
            for i in range(len(self.task_configs)):
                batch = next(iterators[i])
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                raw_seqs = batch['raw_sequences']
                
                debug = (step == 0 and i == 0 and epoch_index == 1)
                
                logits, _ = self.forward(input_ids, mask, raw_seqs, task_idx=i, debug=debug)
                
                is_token = (self.task_configs[i]['type'] == 'token_classification')
                if is_token: loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                else: loss = self.loss_fns[i](logits, targets)
                
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = (precision * loss) + self.log_vars[i]
                
                weighted_loss.backward()
                step_loss += weighted_loss.item()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            epoch_loss += step_loss
            
            if step % 100 == 0:
                self.log_sigmas(epoch_index, step)
                lr = scheduler.get_last_lr()[0] if scheduler else 0
                print(f"  Step {step}/{max_steps} | Loss: {step_loss:.4f} | LR: {lr:.2e}")

        return {"avg_loss": epoch_loss / max_steps}

    def evaluate(self, loader_list=None, split_name="Validation"):
        if loader_list is None: loader_list = self.valid_loaders
        for m in self.experts: m.eval()
        for m in self.heads: m.eval()
        self.bio_router.eval()
        
        raw_results = {
            'Thermo': {'true': [], 'pred': [], 'weights': []},
            'SSP': {'true': [], 'pred': [], 'weights': []},
            'Cloning': {'true': [], 'pred': [], 'prob': [], 'weights': []}
        }
        
        metrics_log = {}
        print(f"\n[{split_name}] Evaluating & Collecting Data...")
        
        with torch.no_grad():
            for i, loader in enumerate(loader_list):
                task_name = self.task_configs[i]['name']
                is_token = (self.task_configs[i]['type'] == 'token_classification')
                
                total_loss = 0; correct = 0; total = 0
                
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    raw_seqs = batch['raw_sequences']
                    
                    logits, weights = self.forward(input_ids, mask, raw_seqs, task_idx=i)
                    batch_mean_weights = weights.mean(dim=0)
                    
                    if i == 0: # Thermo
                        raw_results['Thermo']['true'].extend(targets.view(-1).cpu().numpy())
                        raw_results['Thermo']['pred'].extend(logits.view(-1).cpu().numpy())
                        raw_results['Thermo']['weights'].append(batch_mean_weights.cpu().numpy())
                        loss = self.loss_fns[i](logits, targets)
                        total_loss += loss.item() * input_ids.size(0); total += input_ids.size(0)
                        
                    elif i == 1: # SSP
                        loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                        total_loss += loss.item() * input_ids.size(0); total += input_ids.size(0)
                        p = logits.argmax(dim=-1).view(-1); t = targets.view(-1); m = t!=-100
                        if m.sum()>0: correct += (p[m]==t[m]).sum().item(); total += m.sum().item()
                        raw_results['SSP']['weights'].append(batch_mean_weights.cpu().numpy())

                    elif i == 2: # Cloning
                        loss = self.loss_fns[i](logits, targets)
                        total_loss += loss.item() * input_ids.size(0); total += input_ids.size(0)
                        
                        # --- FIX HERE: CALCULATE ACCURACY ---
                        correct += (logits.argmax(dim=1) == targets).sum().item()
                        # ------------------------------------
                        
                        probs = torch.softmax(logits, dim=1)[:, 1]
                        raw_results['Cloning']['true'].extend(targets.cpu().numpy())
                        raw_results['Cloning']['pred'].extend(logits.argmax(dim=1).cpu().numpy())
                        raw_results['Cloning']['prob'].extend(probs.cpu().numpy())
                        raw_results['Cloning']['weights'].append(batch_mean_weights.cpu().numpy())

                if i == 0: metrics_log[task_name] = f"MSE: {total_loss/total:.4f}"
                else: metrics_log[task_name] = f"Acc: {correct/(total if total>0 else 1):.4f}"
                print(f"  {task_name}: {metrics_log[task_name]}")
                
        return metrics_log, raw_results