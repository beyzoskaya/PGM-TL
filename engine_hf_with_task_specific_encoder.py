import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from itertools import cycle

def multitask_collate_fn(batch, tokenizer):
    """
    Handles batches where inputs are sequences and targets vary by task type.
    Dynamically pads to the longest sequence in the *current batch* to save memory.
    """
    sequences = [item['sequence'] for item in batch]
    
    # 1. Tokenize
    # padding=True: Pads to the longest sequence IN THIS BATCH (efficient)
    # truncation=True: Cuts off if longer than max_length (usually 1024 for ProtBERT)
    inputs = tokenizer(
        sequences, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=1024 
    )
    
    # 2. Process Targets
    # We extract the raw target value from the dictionary structure
    raw_targets = [item['targets']['target'] for item in batch]
    
    target_tensor = None
    
    # --- LOGIC TO DETECT TARGET TYPE ---
    
    # Case A: List of Ints -> Secondary Structure (Token Classification)
    if raw_targets[0] is not None and isinstance(raw_targets[0], list):
        batch_size = len(raw_targets)
        max_seq_len = inputs['input_ids'].shape[1] # Match input length
        
        # Create a tensor filled with -100 (PyTorch's default ignore_index)
        target_tensor = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
        
        for i, t_seq in enumerate(raw_targets):
            # If target is shorter than padded seq, we fill valid parts
            # If target is longer (truncation), we slice it
            valid_len = min(len(t_seq), max_seq_len)
            if valid_len > 0:
                target_tensor[i, :valid_len] = torch.tensor(t_seq[:valid_len], dtype=torch.long)

    # Case B: Float -> Thermostability (Regression)
    elif raw_targets[0] is not None and isinstance(raw_targets[0], float):
        target_tensor = torch.tensor(raw_targets, dtype=torch.float32).unsqueeze(1) # Shape [B, 1]

    # Case C: Int -> Cloning (Sequence Classification)
    elif raw_targets[0] is not None and isinstance(raw_targets[0], int):
        target_tensor = torch.tensor(raw_targets, dtype=torch.long) # Shape [B]
        
    # Case D: Handle None/Missing (Evaluation with no targets)
    else:
        # Fallback placeholder
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
        
        # Optional: Manual weighting of tasks (e.g. [1.0, 5.0, 1.0])
        # Useful if one task's loss is naturally very small compared to others
        self.task_weights = task_weights if task_weights else [1.0] * len(task_configs)

        # --- BUILD TASK HEADS ---
        self.heads = nn.ModuleList()
        self.loss_fns = []
        
        hidden_dim = backbone.hidden_size
        
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                # Thermostability
                self.heads.append(nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.Tanh(), # Tanh often stabilizes regression on protein embeddings
                    nn.Linear(hidden_dim // 2, 1)
                ).to(device))
                self.loss_fns.append(nn.MSELoss())
                
            elif cfg['type'] == 'token_classification' or cfg['type'] == 'per_residue_classification':
                # Secondary Structure
                self.heads.append(nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, cfg['num_labels'])
                ).to(device))
                # ignore_index=-100 means we don't compute loss on padding tokens
                self.loss_fns.append(nn.CrossEntropyLoss(ignore_index=-100))
                
            else: 
                # Sequence Classification (Cloning)
                self.heads.append(nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, cfg['num_labels'])
                ).to(device))
                self.loss_fns.append(nn.CrossEntropyLoss())

        # --- BUILD DATA LOADERS ---
        # We bind the tokenizer to the collate function using lambda
        tokenizer = backbone.tokenizer
        
        self.train_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True, 
                       collate_fn=lambda b: multitask_collate_fn(b, tokenizer))
            for ds in train_sets
        ]
        
        if valid_sets:
            self.valid_loaders = [
                DataLoader(ds, batch_size=batch_size, shuffle=False, 
                           collate_fn=lambda b: multitask_collate_fn(b, tokenizer))
                for ds in valid_sets
            ]
        else:
            self.valid_loaders = None

        if test_sets:
            self.test_loaders = [
                DataLoader(ds, batch_size=batch_size, shuffle=False, 
                           collate_fn=lambda b: multitask_collate_fn(b, tokenizer))
                for ds in test_sets
            ]
        else:
            self.test_loaders = None

    def train_one_epoch(self, optimizer):
        self.backbone.train()
        for h in self.heads: h.train()

        loader_lens = [len(l) for l in self.train_loaders]
        max_steps = max(loader_lens)
        iterators = [cycle(l) for l in self.train_loaders]
        
        epoch_loss = 0
        print(f"\n[Train] Starting epoch with {max_steps} steps (Gradient Accumulation Mode)...")
        
        for step in range(max_steps):
            optimizer.zero_grad()
            step_loss_sum = 0
            
            # --- GRADIENT ACCUMULATION LOOP ---
            for i in range(len(self.task_configs)):
                batch = next(iterators[i])
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)

                # === DEBUG BLOCK (Runs only on first step) ===
                if step == 0:
                    print(f"\n  [DEBUG] Task {i} ({self.task_configs[i].get('name', 'Unknown')})")
                    print(f"   - Input Shape: {input_ids.shape}")
                    print(f"   - Target Shape: {targets.shape}")
                    # Check for shape mismatches specifically for classification
                    if self.task_configs[i]['type'] == 'token_classification':
                        print(f"   - Target Sample: {targets[0, :10].cpu().tolist()} (Should contain -100)")
                # =============================================

                is_token_task = (self.task_configs[i]['type'] in ['token_classification', 'per_residue_classification'])
                task_mode = 'token' if is_token_task else 'sequence'
                
                embeddings = self.backbone(input_ids, mask, task_type=task_mode)
                logits = self.heads[i](embeddings)

                if is_token_task:
                    loss = self.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                else:
                    loss = self.loss_fns[i](logits, targets)

                # === DEBUG LOSS ===
                if step == 0:
                    print(f"   - Raw Loss: {loss.item():.6f}")
                # ==================

                weighted_loss = loss * self.task_weights[i]
                weighted_loss.backward()
                step_loss_sum += weighted_loss.item()

            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += step_loss_sum

            if step % 50 == 0:
                print(f"  Step {step}/{max_steps} | Combined Loss: {step_loss_sum:.4f}")

        return epoch_loss / max_steps

    def evaluate(self, loader_list=None, split_name="Validation"):
        """
        Standard evaluation loop. Returns a dictionary of metrics.
        """
        if loader_list is None:
            loader_list = self.valid_loaders
            
        if not loader_list:
            print("No validation loaders provided.")
            return {}

        self.backbone.eval()
        for h in self.heads: h.eval()
        
        metrics = {}
        print(f"\n[{split_name}] Evaluating...")

        with torch.no_grad():
            for i, loader in enumerate(loader_list):
                task_name = self.task_configs[i].get('name', f'Task_{i}')
                is_token_task = (self.task_configs[i]['type'] in ['token_classification', 'per_residue_classification'])
                is_regression = (self.task_configs[i]['type'] == 'regression')
                
                total_loss = 0
                correct = 0
                total_samples = 0
                
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    task_mode = 'token' if is_token_task else 'sequence'
                    embeddings = self.backbone(input_ids, mask, task_type=task_mode)
                    logits = self.heads[i](embeddings)

                    # METRIC CALCULATION
                    if is_token_task:
                        # Accuracy on valid tokens (ignoring -100)
                        preds = logits.argmax(dim=-1).view(-1)
                        lbls = targets.view(-1)
                        mask_valid = lbls != -100
                        
                        if mask_valid.sum() > 0:
                            correct += (preds[mask_valid] == lbls[mask_valid]).sum().item()
                            total_samples += mask_valid.sum().item()
                            
                    elif is_regression:
                        # MSE Loss
                        loss = self.loss_fns[i](logits, targets)
                        total_loss += loss.item() * input_ids.size(0)
                        total_samples += input_ids.size(0)
                        
                    else: # Classification
                        preds = logits.argmax(dim=1)
                        correct += (preds == targets).sum().item()
                        total_samples += input_ids.size(0)

                # Formatting Results
                if is_regression:
                    mse = total_loss / total_samples if total_samples > 0 else 0
                    metrics[task_name] = {"MSE": float(f"{mse:.4f}")}
                    print(f"  {task_name}: MSE = {mse:.4f}")
                else:
                    acc = correct / total_samples if total_samples > 0 else 0
                    metrics[task_name] = {"Accuracy": float(f"{acc:.4f}")}
                    print(f"  {task_name}: Accuracy = {acc:.4f}")

        return metrics