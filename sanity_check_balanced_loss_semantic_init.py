import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_semantic_prompt_tuning import TaskPromptedEngine

def print_section(title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")

def run_debug():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Debug on: {device}")

    print_section("1. LOADING MINI DATASETS")
    try:
        ds_t = Thermostability(verbose=0)
        ds_s = SecondaryStructure(verbose=0)
        ds_c = CloningCLF(verbose=0)
        
        # Take 16 samples (1 batch) for each to properly test batch processing
        train_sets = [Subset(ds_t, range(16)), Subset(ds_s, range(16)), Subset(ds_c, range(16))]
        valid_sets = [Subset(ds_t, range(16, 32)), Subset(ds_s, range(16, 32)), Subset(ds_c, range(16, 32))]
        
        print("✓ Subsets prepared (16 samples each).")
    except Exception as e:
        print(f"❌ Data Load Error: {e}")
        return

    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    print_section("2. INITIALIZING BALANCED ENGINE")
    print("Loading Backbone (Rank=2 for speed)...")
    # Using Unfrozen=0 and Rank=2 just to make it fast for debugging
    backbone = SharedProtBert(lora_rank=2, unfrozen_layers=0)
    
    print("Initializing Engine with 'semantic' strategy...")
    engine = TaskPromptedEngine(
        backbone=backbone, 
        task_configs=task_configs,
        train_sets=train_sets, 
        valid_sets=valid_sets,
        batch_size=4, # Small batch for debug
        device=device,
        init_strategy="semantic"
    )

    print_section("3. TESTING BALANCED TRAINING STEP")
    
    optimizer = torch.optim.AdamW(engine.parameters(), lr=1e-4)
    
    print("Running 'train_one_epoch' (Will stop after a few steps)...")

    try:
        # We assume the engine code prints the table at the start
        # We just want to see if it crashes during the forward/backward pass
        engine.backbone.train()
        
        # Manually trigger the logic that happens inside train_one_epoch
        # to ensure scaling tensors are on the correct device
        loader_lens = [len(l) for l in engine.train_loaders]
        max_steps = max(loader_lens)
        loss_scales = [length / max_steps for length in loader_lens]
        
        print(f"Calculated Scales: {loss_scales}")
        print("✓ Loss Scale calculation works.")
        
        print("\nAttempting 1 Real Optimization Step...")
        res = engine.train_one_epoch(optimizer, None, epoch_index=1)
        print("✓ Train Step completed without crashing.")
        
    except Exception as e:
        print(f"❌ Training Logic Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print_section("4. TESTING NEW METRICS (Spearman, F1, MCC)")
    
    try:
        metrics, _ = engine.evaluate(loader_list=engine.valid_loaders, split_name="DEBUG_VAL")
        
        print("\nExtracted Metrics:")
        for task, vals in metrics.items():
            print(f"  [{task}]")
            for k, v in vals.items():
                print(f"    - {k}: {v}")
                
        if 'Spearman' in metrics['Thermostability']:
            print("  ✅ Spearman Calculated.")
        if 'F1' in metrics['SecStructure']:
            print("  ✅ F1 Calculated.")
        if 'MCC' in metrics['Cloning']:
            print("  ✅ MCC Calculated.")
            
    except Exception as e:
        print(f"❌ Evaluation/Metric Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print_section("DEBUG PASSED - READY FOR ORIGINAL TRAINING")

if __name__ == "__main__":
    run_debug()