import torch
import torch.nn as nn
from torch.utils.data import Subset
import numpy as np

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBertPromptTuning
from engine_hf_prompt import TaskPromptedEngine

def print_section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def run_debug_integration():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_section(f"1. LOADING REAL DATA SUBSETS (Device: {device})")

    try:
        print("  [Data] Loading Thermostability...")
        ds_thermo = Thermostability(verbose=0)
        sub_thermo = Subset(ds_thermo, range(4))

        print("  [Data] Loading SecondaryStructure...")
        ds_ssp = SecondaryStructure(verbose=0)
        sub_ssp = Subset(ds_ssp, range(4))

        print("  [Data] Loading CloningCLF...")
        ds_cloning = CloningCLF(verbose=0)
        sub_cloning = Subset(ds_cloning, range(4))
        
        print("  ✓ Subsets created successfully.")
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return

    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    print_section("2. INITIALIZING ENGINE WITH SUBSETS")
    
    print("  [Model] Loading Backbone (SharedProtBert)...")
    backbone = SharedProtBertPromptTuning(lora_rank=2, unfrozen_layers=0)
    
    print("  [Engine] Initializing TaskPromptedEngine...")
    engine = TaskPromptedEngine(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=[sub_thermo, sub_ssp, sub_cloning], # <--- Real Data
        valid_sets=None, # Not needed for this check
        batch_size=2,    # Small batch size for reading logs easily
        device=device
    )
    
    print_section("3. RUNNING FORWARD PASS CHECKS")
    
    for i, loader in enumerate(engine.train_loaders):
        task_name = task_configs[i]['name']
        task_type = task_configs[i]['type']
        
        print(f"\n>>> CHECKING TASK {i}: {task_name} ({task_type}) <<<")
        
        # A. Fetch Batch (Tests Collate Function)
        try:
            batch = next(iter(loader))
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            print(f"  [Collate] Batch successfully loaded.")
            print(f"  [Collate] Input Shape: {input_ids.shape}")
            print(f"  [Collate] Target Shape: {targets.shape}")
            
            # Print raw target example to ensure normalization/formatting is sane
            if task_type == 'regression':
                print(f"  [Targets] Sample: {targets[0].item():.4f}")
            elif task_type == 'token_classification':
                print(f"  [Targets] Sample Length: {(targets[0] != -100).sum()} valid tokens")
                
        except Exception as e:
            print(f"  ❌ CRITICAL FAIL in Data Loading: {e}")
            continue

        # B. Forward Pass (Tests Prompt Injection & Slicing)
        print(f"  [Forward] Calling engine(..., debug=True)...")
        print("-" * 20 + " INTERNAL DEBUG LOGS " + "-" * 20)
        
        # This will trigger the prints inside protbert_hf.py and engine_hf_prompt.py
        logits = engine(input_ids, mask, task_idx=i, debug=True)
        
        print("-" * 60)
        print(f"  [Output] Final Logits Shape: {logits.shape}")

        # C. Verification Logic
        seq_len_input = input_ids.shape[1]
        
        if task_type == 'token_classification':
            # SSP: Shape should be [Batch, SeqLen, 8]
            # It must NOT be SeqLen + 1 (which would happen if prompt wasn't removed)
            if logits.shape[1] == seq_len_input:
                print("  ✅ SUCCESS: Prompt token correctly sliced off before output.")
            else:
                print(f"  ❌ FAIL: Output length {logits.shape[1]} != Input {seq_len_input}. Prompt might be leaking!")
        else:
            # Pooling: Shape should be [Batch, NumLabels] (2D)
            if len(logits.shape) == 2:
                print("  ✅ SUCCESS: Sequence correctly pooled.")
            else:
                print(f"  ❌ FAIL: Expected 2D pooling output, got {logits.shape}")

        # D. Backward Pass (Tests Gradient Flow)
        print(f"  [Backward] Checking Gradient Flow...")
        loss_fn = engine.loss_fns[i]
        
        # Calculate Dummy Loss
        if task_type == 'token_classification':
            loss = loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
        else:
            loss = loss_fn(logits, targets)
            
        engine.zero_grad()
        loss.backward()
        
        # Check if the prompt for THIS task got updated
        prompt_grad = engine.task_prompts[i].grad
        if prompt_grad is not None:
            norm = prompt_grad.norm().item()
            print(f"  [Grads] Task Prompt Gradient Norm: {norm:.8f}")
            if norm > 0:
                print(f"  ✅ SUCCESS: {task_name} Prompt is learning.")
            else:
                print(f"  ⚠️ WARNING: Gradient is zero.")
        else:
            print(f"  ❌ FAIL: No gradient reached the task prompt.")

    print_section("DEBUG COMPLETE")

if __name__ == "__main__":
    run_debug_integration()