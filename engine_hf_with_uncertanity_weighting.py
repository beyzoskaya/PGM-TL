import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from itertools import cycle
import os
import csv
import datetime

def multitask_collate_fn(batch, tokenizer):
    """Same collate function as before"""
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

        # --- NOVELTY 1: Learnable Log-Variance Parameters ---
        # Initialize at 0.0 (Sigma=1.0)
        self.log_vars = nn.Parameter(torch.zeros(len(task_configs), device=device))

        # --- LOGGING SETUP ---
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. CSV for Sigmas (Plotting)
        self.sigma_log_path = os.path.join(save_dir, "training_dynamics_sigmas.csv")
        if not os.path.exists(self.sigma_log_path):
            with open(self.sigma_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["Epoch", "Step"] + [f"Sigma_{cfg['name']}" for cfg in task_configs] + [f"Loss_{cfg['name']}" for cfg in task_configs]
                writer.writerow(header)

        # 2. Text file for Gradient Analysis (Inspection)
        self.grad_log_path = os.path.join(save_dir, "gradient_conflict_log.txt")
        with open(self.grad_log_path, 'a') as f:
            f.write(f"\n\n--- NEW TRAINING RUN STARTED: {datetime.datetime.now()} ---\n")

        # --- HEADS & LOSSES ---
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

        # Loaders
        tokenizer = backbone.tokenizer
        self.train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in train_sets]
        self.valid_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in valid_sets] if valid_sets else None
        self.test_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in test_sets] if test_sets else None

    def analyze_gradients(self, losses, step, epoch):
        """
        Computes Cosine Similarity and logs it to file.
        """
        grads = []
        trainable_params = [p for p in self.backbone.parameters() if p.requires_grad]
        
        if not trainable_params: 
            return # Nothing to analyze if backbone is totally frozen (shouldn't happen with LoRA)

        # 1. Compute gradients
        for loss in losses:
            g = torch.autograd.grad(loss, trainable_params, retain_graph=True, allow_unused=True)
            flat_g = torch.cat([x.flatten() for x in g if x is not None])
            grads.append(flat_g)
        
        # 2. Compute Similarity & Log
        log_msg = f"\n[Epoch {epoch} Step {step}] Gradient Cosine Similarity:\n"
        num_tasks = len(grads)
        
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                sim = F.cosine_similarity(grads[i].unsqueeze(0), grads[j].unsqueeze(0))
                t1 = self.task_configs[i]['name']
                t2 = self.task_configs[j]['name']
                log_msg += f"  > {t1} vs {t2}: {sim.item():.4f}\n"
        
        # Write to file
        with open(self.grad_log_path, 'a') as f:
            f.write(log_msg)
        
        # Also print to console for monitoring
        print(log_msg)

    def log_dynamics(self, epoch, step, losses):
        """
        Saves current Sigma values and Raw Losses to CSV
        """
        sigmas = torch.exp(self.log_vars).detach().cpu().numpy().tolist()
        raw_losses = [l.item() for l in losses]
        
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
            raw_losses_list = []
            
            # Analyze gradients occasionally (e.g., step 50, then every 500 steps)
            should_analyze_grads = (step == 50) or (step % 500 == 0 and step > 0)
            
            # Log sigmas more frequently (every 100 steps)
            should_log_sigmas = (step % 100 == 0)

            for i in range(len(self.task_configs)):
                batch = next(iterators[i])
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)

                is_token_task = (self.task_configs[i]['type'] in ['token_classification', 'per_residue_classification'])
                
                embeddings = self.backbone(input_ids, mask, task_type='token' if is_token_task else 'sequence')
                logits = self.heads[i](embeddings)

                if is_token_task:
                    loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                else:
                    loss = self.loss_fns[i](logits, targets)
                
                # Store raw loss for logging/analysis
                raw_losses_list.append(loss)

                # Uncertainty Weighting Formula
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = (precision * loss) + self.log_vars[i]
                
                weighted_loss.backward()
                step_loss_sum += weighted_loss.item()
                
                # Stats accumulation
                with torch.no_grad():
                    batch_size = input_ids.size(0)
                    task_stats[i]['loss'] += loss.item() * batch_size
                    if self.task_configs[i]['type'] == 'regression':
                        task_stats[i]['total'] += batch_size
                    elif is_token_task:
                        preds = logits.argmax(dim=-1).view(-1)
                        lbls = targets.view(-1)
                        mask_valid = lbls != -100
                        if mask_valid.sum() > 0:
                            task_stats[i]['correct'] += (preds[mask_valid] == lbls[mask_valid]).sum().item()
                            task_stats[i]['total'] += mask_valid.sum().item()
                    else:
                        preds = logits.argmax(dim=1)
                        task_stats[i]['correct'] += (preds == targets).sum().item()
                        task_stats[i]['total'] += batch_size

            # --- SAVINGS & ANALYSIS ---
            if should_analyze_grads:
                self.analyze_gradients(raw_losses_list, step, epoch_index)
            
            if should_log_sigmas:
                self.log_dynamics(epoch_index, step, raw_losses_list)

            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
            optimizer.step()
            epoch_loss += step_loss_sum

            if step % 50 == 0:
                # Just for console visibility
                sigmas_str = [f"{np.exp(v):.2f}" for v in self.log_vars.detach().cpu().numpy()]
                print(f"  Step {step}/{max_steps} | Combined Loss: {step_loss_sum:.4f} | Sigmas: {sigmas_str}")

        # Return metrics (same as before)
        results = {"combined_loss": epoch_loss / max_steps}
        for i, cfg in enumerate(self.task_configs):
            name = cfg['name']
            avg_loss = task_stats[i]['loss'] / max(task_stats[i]['total'], 1)
            if cfg['type'] == 'regression':
                results[name] = f"MSE: {avg_loss:.4f}"
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
                
                total_loss = 0
                correct = 0
                total_samples = 0
                total_tokens = 0
                
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    embeddings = self.backbone(input_ids, mask, task_type='token' if is_token else 'sequence')
                    logits = self.heads[i](embeddings)

                    if is_token:
                        loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                        total_loss += loss.item() * input_ids.size(0)
                        total_samples += input_ids.size(0)
                        preds = logits.argmax(dim=-1).view(-1)
                        lbls = targets.view(-1)
                        mask_valid = lbls != -100
                        if mask_valid.sum() > 0:
                            correct += (preds[mask_valid] == lbls[mask_valid]).sum().item()
                            total_tokens += mask_valid.sum().item()

                    elif is_reg:
                        loss = self.loss_fns[i](logits, targets)
                        total_loss += loss.item() * input_ids.size(0)
                        total_samples += input_ids.size(0)
                    else:
                        loss = self.loss_fns[i](logits, targets)
                        total_loss += loss.item() * input_ids.size(0)
                        total_samples += input_ids.size(0)
                        preds = logits.argmax(dim=1)
                        correct += (preds == targets).sum().item()

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