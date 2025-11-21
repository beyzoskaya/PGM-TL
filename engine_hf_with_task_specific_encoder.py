import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from itertools import cycle
import re

def multitask_collate_fn(batch, tokenizer):
    """
    Handles batches where inputs are sequences and targets vary by task type.
    Dynamically pads to the longest sequence in the *current batch*.
    FIX: Adds whitespace between amino acids for ProtBERT tokenizer.
    """
    # 1. Prepare Sequences (Add spaces: "MKTV" -> "M K T V")
    sequences = []
    for item in batch:
        seq = item['sequence']
        if isinstance(seq, str):
            sequences.append(" ".join(list(seq)))
        else:
            sequences.append("")
    
    # 2. Tokenize
    inputs = tokenizer(
        sequences, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=1024 
    )
    
    # 3. Process Targets
    raw_targets = [item['targets']['target'] for item in batch]
    target_tensor = None
    
    # Case A: Secondary Structure (List of Ints)
    if raw_targets[0] is not None and isinstance(raw_targets[0], list):
        batch_size = len(raw_targets)
        max_seq_len = inputs['input_ids'].shape[1] 
        
        # Fill with -100
        target_tensor = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
        
        for i, t_seq in enumerate(raw_targets):
            # Align targets to tokens (Skip [CLS] at index 0)
            valid_len = min(len(t_seq), max_seq_len - 2) 
            if valid_len > 0:
                target_tensor[i, 1 : 1+valid_len] = torch.tensor(t_seq[:valid_len], dtype=torch.long)

    # Case B: Thermostability (Float)
    elif raw_targets[0] is not None and isinstance(raw_targets[0], float):
        target_tensor = torch.tensor(raw_targets, dtype=torch.float32).unsqueeze(1)

    # Case C: Cloning (Int)
    elif raw_targets[0] is not None and isinstance(raw_targets[0], int):
        target_tensor = torch.tensor(raw_targets, dtype=torch.long)
        
    else:
        target_tensor = torch.zeros(len(raw_targets))

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'targets': target_tensor
    }

class MultiTaskEngine(nn.Module):
    def __init__(self, backbone, task_configs, train_sets, valid_sets, 
                 test_sets=None, batch_size=8, device='cuda', task_weights=None):
        super().__init__()
        self.backbone = backbone.to(device)
        self.device = device
        self.task_configs = task_configs
        self.batch_size = batch_size
        self.task_weights = task_weights if task_weights else [1.0] * len(task_configs)

        # --- BUILD TASK HEADS ---
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

        # --- BUILD DATA LOADERS ---
        tokenizer = backbone.tokenizer
        self.train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in train_sets]
        self.valid_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in valid_sets] if valid_sets else None
        self.test_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: multitask_collate_fn(b, tokenizer)) for ds in test_sets] if test_sets else None

    def train_one_epoch(self, optimizer):
        self.backbone.train()
        for h in self.heads: h.train()

        loader_lens = [len(l) for l in self.train_loaders]
        max_steps = max(loader_lens)
        iterators = [cycle(l) for l in self.train_loaders]
        
        epoch_loss = 0
        task_stats = {i: {'loss': 0.0, 'total': 0, 'correct': 0} for i in range(len(self.task_configs))}
        
        print(f"\n[Train] Starting epoch with {max_steps} steps...")
        
        for step in range(max_steps):
            optimizer.zero_grad()
            step_loss_sum = 0
            
            for i in range(len(self.task_configs)):
                batch = next(iterators[i])
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)

                is_token_task = (self.task_configs[i]['type'] in ['token_classification', 'per_residue_classification'])
                task_mode = 'token' if is_token_task else 'sequence'
                
                embeddings = self.backbone(input_ids, mask, task_type=task_mode)
                logits = self.heads[i](embeddings)

                if is_token_task:
                    loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                else:
                    loss = self.loss_fns[i](logits, targets)

                weighted_loss = loss * self.task_weights[i]
                weighted_loss.backward()
                step_loss_sum += weighted_loss.item()
                
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

            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
            optimizer.step()
            epoch_loss += step_loss_sum

            if step % 50 == 0:
                print(f"  Step {step}/{max_steps} | Combined Loss: {step_loss_sum:.4f}")

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
        
        print(f"\n[{split_name}] Evaluating...")

        with torch.no_grad():
            for i, loader in enumerate(loader_list):
                task_name = self.task_configs[i].get('name', f'Task_{i}')
                is_token = (self.task_configs[i]['type'] in ['token_classification', 'per_residue_classification'])
                is_reg = (self.task_configs[i]['type'] == 'regression')
                
                total_loss = 0
                correct = 0
                total_samples = 0
                total_tokens = 0 # For token accuracy specifically
                
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    embeddings = self.backbone(input_ids, mask, task_type='token' if is_token else 'sequence')
                    logits = self.heads[i](embeddings)

                    # Calculate Loss & Metrics
                    if is_token:
                        # Loss
                        loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                        total_loss += loss.item() * input_ids.size(0)
                        total_samples += input_ids.size(0)
                        
                        # Acc
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

                # Print Results
                avg_loss = total_loss / total_samples if total_samples > 0 else 0
                
                if is_reg:
                    print(f"  {task_name}: MSE (Loss) = {avg_loss:.4f}")
                elif is_token:
                    acc = correct / total_tokens if total_tokens > 0 else 0
                    print(f"  {task_name}: Loss = {avg_loss:.4f} | Accuracy = {acc:.4f}")
                else:
                    acc = correct / total_samples if total_samples > 0 else 0
                    print(f"  {task_name}: Loss = {avg_loss:.4f} | Accuracy = {acc:.4f}")