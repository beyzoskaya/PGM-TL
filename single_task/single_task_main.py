import os
import torch
import numpy as np
import logging

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engines.engine_hf_with_task_specific_encoder import MultiTaskEngine

TARGET_TASK = 'Cloning' 

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LORA_RANK = 16
UNFROZEN_LAYERS = 2
#UNFROZEN_LAYERS = 0 #FIXME: Try freezing all except LoRA for both single task and multi task 
SAVE_DIR = f"/content/drive/MyDrive/protein_multitask_outputs/baseline_{TARGET_TASK}"
os.makedirs(SAVE_DIR, exist_ok=True)

# Epochs: Small datasets need more epochs to converge than the huge SSP dataset
if TARGET_TASK == 'SecStructure':
    EPOCHS = 5  
else:
    EPOCHS = 50  

LEARNING_RATE = 1e-4

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_thermo(train_ds, valid_ds, test_ds):
    print(f"\n[Data] Normalizing {TARGET_TASK} targets...")
    full_dataset = train_ds.dataset 
    train_indices = train_ds.indices
    all_raw_targets = full_dataset.targets['target']
    
    train_values = [all_raw_targets[i] for i in train_indices if all_raw_targets[i] is not None]
    mean = np.mean(train_values)
    std = np.std(train_values)
    print(f"  Mean: {mean:.4f} | Std: {std:.4f}")

    new_targets = []
    for t in all_raw_targets:
        if t is None: new_targets.append(None)
        else: new_targets.append((t - mean) / std)
    full_dataset.targets['target'] = new_targets
    print("  ✓ Normalization complete.")

def main():
    set_seed(SEED)
    print(f"==================================================")
    print(f"SINGLE TASK BASELINE: {TARGET_TASK}")
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"==================================================\n")

    # 1. LOAD ONLY THE TARGET DATASET
    print("[1/5] Loading Data...")
    
    task_config = {}
    train_set, valid_set, test_set = None, None, None

    if TARGET_TASK == 'Thermostability':
        ds = Thermostability(verbose=0)
        train_set, valid_set, test_set = ds.split()
        normalize_thermo(train_set, valid_set, test_set)
        task_config = {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1}

    elif TARGET_TASK == 'SecStructure':
        ds = SecondaryStructure(verbose=0)
        train_set, valid_set, test_set = ds.split()
        task_config = {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8}

    elif TARGET_TASK == 'Cloning':
        ds = CloningCLF(verbose=0)
        train_set, valid_set, test_set = ds.split()
        task_config = {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    
    else:
        raise ValueError(f"Unknown task: {TARGET_TASK}")

    print(f"  Train Size: {len(train_set)}")
    print(f"  Valid Size: {len(valid_set)}")

    # 2. INIT MODEL (Exact same architecture as Multi-Task)
    print(f"\n[2/5] Initializing SharedProtBert...")
    backbone = SharedProtBert(
        lora_rank=LORA_RANK,
        lora_dropout=0.1,
        unfrozen_layers=UNFROZEN_LAYERS
    )

    # 3. INIT ENGINE
    # Note: We pass lists with 1 element to reuse the existing robust Engine code
    print(f"\n[3/5] Initializing Engine...")
    engine = MultiTaskEngine(
        backbone=backbone,
        task_configs=[task_config],    # List of 1
        train_sets=[train_set],        # List of 1
        valid_sets=[valid_set],        # List of 1
        test_sets=[test_set],          # List of 1
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    optimizer = torch.optim.AdamW(engine.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 4. TRAINING LOOP
    print(f"\n[4/5] Starting Single-Task Training ({EPOCHS} epochs)...")
    
    best_val_score = float('inf') if task_config['type'] == 'regression' else float('-inf')
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*10} EPOCH {epoch+1}/{EPOCHS} {'='*10}")
        
        # Train
        metrics = engine.train_one_epoch(optimizer)
        
        # Print simplified metrics
        print(f"  [Train] Loss: {metrics['combined_loss']:.4f}")
        
        # Validate
        val_metrics = engine.evaluate(loader_list=engine.valid_loaders, split_name="VALIDATION")
        
        # Extract the main metric for saving best model
        current_val_score = 0.0
        metric_str = ""
        
        # Parse the output dictionary
        for k, v in val_metrics.items(): # There is only one key
            # val_metrics format is usually {TaskName: "Loss: X | Acc: Y"} or "MSE: X"
            # We rely on the print output mostly, but for logic:
            pass 

        # Save Checkpoint (Always save latest, track best in your head or logs)
        save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save(engine.state_dict(), save_path)
        print(f"  >> Saved checkpoint: {save_path}")

    print("\n[5/5] Final Test Evaluation...")
    engine.evaluate(loader_list=engine.test_loaders, split_name="TEST")

if __name__ == "__main__":
    main()