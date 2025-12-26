import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os

from engine_hf_prompt import TaskPromptedEngine
from protbert_hf import SharedProtBert

class MockDataset(Dataset):
    def __init__(self, task_type, length=10):
        self.sequences = ["MACDEFGHIKLMNPQRSTVWY"[:random.randint(5, 20)] for _ in range(length)]
        self.task_type = task_type
        
    def __len__(self): return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if self.task_type == 'ssp':
            # Target length must match sequence length for SSP
            return {'sequence': seq, 'targets': {'target': [random.randint(0,7) for _ in seq]}}
        elif self.task_type == 'thermo':
            return {'sequence': seq, 'targets': {'target': random.random()}}
        elif self.task_type == 'cloning':
            return {'sequence': seq, 'targets': {'target': random.randint(0,1)}}

# --- 2. VERIFICATION SCRIPT ---
def run_debug():
    print("=== STARTING DEBUG PROTOCOL ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # A. Setup Mock Data
    print("\n[Step 1] Creating Mock Dataloaders...")
    train_sets = [
        MockDataset('thermo'),
        MockDataset('ssp'),
        MockDataset('cloning')
    ]
    # Note: Valid/Test not needed for debug logic check
    
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    
    # B. Initialize Tiny Model
    print("\n[Step 2] Initializing Backbone (Freezing check)...")
    # Using small rank to be fast, unfrozen_layers=1 to check grads
    backbone = SharedProtBert(lora_rank=4, unfrozen_layers=1) 
    
    # Verify Freezing
    frozen_count = 0
    grad_count = 0
    for n, p in backbone.named_parameters():
        if p.requires_grad: grad_count += 1
        else: frozen_count += 1
    print(f"   Params with grad: {grad_count} (Should be > 0 due to LoRA + Unfrozen)")
    print(f"   Frozen params: {frozen_count}")

    # C. Initialize Engine
    print("\n[Step 3] Initializing Engine...")
    engine = TaskPromptedEngine(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=train_sets,
        valid_sets=None,
        batch_size=2, # Small batch
        device=device,
        save_dir="./debug_output"
    )
    
    optimizer = torch.optim.AdamW(engine.parameters(), lr=1e-4)
    
    # D. Forward Pass & Prompt Logic Check
    print("\n[Step 4] Running Forward Pass & Shape Check...")
    engine.train()
    
    # We manually pull a batch to debug internals
    tokenizer = backbone.tokenizer
    
    for i, cfg in enumerate(task_configs):
        print(f"   --- Debugging Task: {cfg['name']} ---")
        loader = engine.train_loaders[i]
        batch = next(iter(loader))
        
        inputs = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        
        print(f"   Input Shape: {inputs.shape}")
        
        # 1. Run Backbone manually to see shape
        backbone_out = engine.backbone(inputs, mask, task_type='token')
        print(f"   Backbone Output: {backbone_out.shape}")
        
        # 2. Run Engine Forward
        logits = engine(inputs, mask, task_idx=i, debug=True)
        print(f"   Logits Shape: {logits.shape}")
        
        # 3. Calculate Loss
        if cfg['type'] == 'token_classification':
            loss = engine.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
        else:
            loss = engine.loss_fns[i](logits, targets)
            
        print(f"   Loss: {loss.item():.4f}")
        
        # 4. Backward (Check if Prompt gets gradients)
        loss.backward()
        prompt_grad = engine.task_prompts[i].grad
        
        if prompt_grad is not None:
            grad_norm = prompt_grad.norm().item()
            print(f"   Prompt Grad Norm: {grad_norm:.6f} (Should be > 0)")
            if grad_norm == 0:
                print("   WARNING: Prompt is not learning!")
        else:
            print("   ERROR: Prompt has no gradient!")
            
        optimizer.zero_grad()

    # E. PCGrad Check
    print("\n[Step 5] Simulating PCGrad Conflict...")
    # Create two fake gradients that conflict (180 degrees opposite)
    g1 = torch.tensor([1.0, 1.0], device=device)
    g2 = torch.tensor([-1.0, -1.0], device=device) # Perfectly conflicting
    
    grads = [g1, g2]
    projected, conflicts = engine._project_conflicting(grads)
    
    print(f"   Original Grads: {g1}, {g2}")
    print(f"   Projected Sum: {projected}")
    print(f"   Conflicts Detected: {conflicts}")
    
    if conflicts > 0 and projected.abs().sum() < 0.1:
         print("   SUCCESS: PCGrad neutralized conflicting gradients.")
    elif conflicts > 0:
         print("   SUCCESS: PCGrad detected conflict and modified gradients.")
    else:
         print("   FAILURE: PCGrad logic failed.")

    print("\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    run_debug()