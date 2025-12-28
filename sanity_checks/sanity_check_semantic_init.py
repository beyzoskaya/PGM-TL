import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_semantic_prompt_tuning import TaskPromptedEngine, multitask_collate_fn

def print_section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def run_debug_full():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- STEP 1: DATASET COUNT CHECK ---
    print_section("1. DATASET INTEGRITY CHECK")
    
    datasets = {
        'Thermostability': Thermostability(verbose=0),
        'SecondaryStructure': SecondaryStructure(verbose=0),
        'CloningCLF': CloningCLF(verbose=0)
    }
    
    # We will pick 1 sample from each to run forward pass later
    one_sample_subsets = []
    
    for name, ds in datasets.items():
        # Force load splits to get counts
        splits = ds.split() # [Train, Val, Test]
        print(f"  [{name}]")
        print(f"     - Train Samples: {len(splits[0]):,}")
        print(f"     - Valid Samples: {len(splits[1]):,}")
        print(f"     - Test Samples:  {len(splits[2]):,}")
        
        # Store a subset of size 1 for debugging
        one_sample_subsets.append(Subset(ds, range(1)))

    # --- STEP 2: SEMANTIC INIT VERIFICATION ---
    print_section("2. SEMANTIC INITIALIZATION CHECK")
    
    print("  [Model] Loading Backbone (Rank=2 for speed)...")
    backbone = SharedProtBert(lora_rank=2, unfrozen_layers=0)
    backbone.to(device)
    
    # Get the TRUE [CLS] embedding manually
    print("  [Check] Extracting ground truth [CLS] embedding...")
    true_cls = backbone.get_cls_embedding(device)
    print(f"     - Shape: {true_cls.shape}")
    print(f"     - First 5 values: {true_cls[0, :5].detach().cpu().numpy()}")
    
    # Initialize Engine with 'semantic' strategy
    print("\n  [Engine] Initializing with init_strategy='semantic'...")
    task_configs = [
        {'name': 'Thermo', 'type': 'regression', 'num_labels': 1},
        {'name': 'SSP', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    
    engine = TaskPromptedEngine(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=one_sample_subsets, # Real data, 1 sample each
        valid_sets=None,
        init_strategy="semantic",
        device=device
    )
    
    # Compare Engine Prompt vs True CLS
    print("\n  [Verification] Comparing Task 0 Prompt to [CLS]...")
    prompt_vec = engine.task_prompts[0].data[0, 0, :] # Shape [1024]
    
    # We added random noise (std=0.001), so difference should be small but not zero
    diff = (prompt_vec - true_cls.squeeze()).abs().mean().item()
    print(f"     - Mean Absolute Difference: {diff:.8f}")
    
    if 0.0 < diff < 0.01:
        print("  ✅ SUCCESS: Prompt matches [CLS] (with slight noise).")
    else:
        print("  ❌ FAIL: Prompt is either purely random (diff too high) or identical (diff=0).")

    # --- STEP 3: FORWARD PASS TRACE (ONE SAMPLE) ---
    print_section("3. SINGLE SEQUENCE PROCESSING TRACE")
    
    tokenizer = backbone.tokenizer
    
    # We use engine.train_loaders because we passed the one_sample_subsets there
    for i, loader in enumerate(engine.train_loaders):
        name = task_configs[i]['name']
        print(f"\n>>> TRACING TASK: {name} <<<")
        
        batch = next(iter(loader))
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        
        print(f"  [Input] Sequence String: {datasets['Thermostability' if i==0 else 'SecondaryStructure' if i==1 else 'CloningCLF'].sequences[0][:20]}...")
        
        # Trigger debug prints in Engine
        logits = engine(input_ids, mask, task_idx=i, debug=True)
        
        print(f"  [Output] Final Shape: {logits.shape}")

    print_section("DEBUG COMPLETE")

if __name__ == "__main__":
    run_debug_full()