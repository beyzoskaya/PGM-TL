import os
import torch
import numpy as np
import logging

# Import your modules
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_with_task_specific_encoder import MultiTaskEngine

# --- CONFIGURATION ---
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16        
EPOCHS = 10
LEARNING_RATE = 1e-4   # 1e-4 is good for LoRA + Top Layers Unfrozen
SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1"
os.makedirs(SAVE_DIR, exist_ok=True)

# Model Config
LORA_RANK = 16
UNFROZEN_LAYERS = 2    # Unfreeze top 2 Transformer layers + Pooler

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_regression_targets(train_ds, valid_ds, test_ds):
    """
    CRITICAL: Z-score normalize regression targets (Thermostability).
    Without this, MSE loss (~2500) will crush Classification loss (~0.7).
    """
    print("\n[Data] Normalizing Thermostability targets...")
    
    # 1. Collect training values to compute stats (avoid data leakage)
    train_values = [x for x in train_ds.targets['target'] if x is not None]
    mean = np.mean(train_values)
    std = np.std(train_values)
    
    print(f"  Mean: {mean:.4f} | Std: {std:.4f}")
    
    # 2. Helper to apply normalization
    def apply_norm(ds):
        new_targets = []
        for t in ds.targets['target']:
            if t is None:
                new_targets.append(None)
            else:
                new_targets.append((t - mean) / std)
        ds.targets['target'] = new_targets

    # 3. Apply to all splits
    apply_norm(train_ds)
    apply_norm(valid_ds)
    apply_norm(test_ds)
    print("  ✓ Normalization complete.")

def main():
    set_seed(SEED)
    print(f"Running on {DEVICE} with Batch Size {BATCH_SIZE}")

    # ====================================================
    # 1. DATA LOADING
    # ====================================================
    print("\n[1/5] Loading Datasets...")
    
    # A. Thermostability (Regression)
    # We load the full wrapper, then get splits
    ds_thermo = Thermostability(verbose=0) 
    thermo_train, thermo_valid, thermo_test = ds_thermo.split()
    # APPLY NORMALIZATION MANUALLY TO BE SAFE
    normalize_regression_targets(thermo_train, thermo_valid, thermo_test)

    # B. Secondary Structure (Token Classification)
    ds_ssp = SecondaryStructure(verbose=0)
    ssp_train, ssp_valid, ssp_test = ds_ssp.split()

    # C. Cloning (Sequence Classification)
    ds_clf = CloningCLF(verbose=0)
    clf_train, clf_valid, clf_test = ds_clf.split()

    print(f"  Thermo Train: {len(thermo_train)} | SSP Train: {len(ssp_train)} | Cloning Train: {len(clf_train)}")

    # Group them for the Engine
    train_sets = [thermo_train, ssp_train, clf_train]
    valid_sets = [thermo_valid, ssp_valid, clf_valid]
    test_sets  = [thermo_test,  ssp_test,  clf_test]

    # ====================================================
    # 2. CONFIGURATION
    # ====================================================
    # The order here MUST match the order in train_sets
    task_configs = [
        {
            'name': 'Thermostability',
            'type': 'regression',
            'num_labels': 1
        },
        {
            'name': 'SecStructure',
            'type': 'token_classification',
            'num_labels': 8 # Q8 Accuracy
        },
        {
            'name': 'Cloning',
            'type': 'sequence_classification',
            'num_labels': 2 # Binary
        }
    ]

    # ====================================================
    # 3. MODEL INITIALIZATION
    # ====================================================
    print(f"\n[2/5] Initializing SharedProtBert (LoRA + Top {UNFROZEN_LAYERS} Unfrozen)...")
    
    backbone = SharedProtBert(
        model_name="Rostlab/prot_bert_bfd",
        lora_rank=LORA_RANK,
        lora_dropout=0.1,
        unfrozen_layers=UNFROZEN_LAYERS
    )

    # ====================================================
    # 4. ENGINE SETUP
    # ====================================================
    print("\n[3/5] Setting up MultiTask Engine...")
    
    # Optional: task_weights=[1.0, 1.0, 1.0]
    # If SSP dominates, you can lower its weight e.g., [1.0, 0.5, 1.0]
    engine = MultiTaskEngine(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=train_sets,
        valid_sets=valid_sets,
        test_sets=test_sets,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    # Optimizer
    # engine.parameters() includes both the Backbone and the Task Heads
    optimizer = torch.optim.AdamW(engine.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    print("\n[Pre-Flight Check] Verifying parameter freezing...")
    frozen_params = 0
    trainable_params = 0
    
    # Check Backbone
    for name, param in backbone.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
            
    print(f"  Backbone Frozen: {frozen_params:,}")
    print(f"  Backbone Trainable: {trainable_params:,} (Should be > 0 due to LoRA)")
    
    if trainable_params == 0:
        raise ValueError("CRITICAL ERROR: No parameters in backbone are trainable!")
        
    # Check Heads
    head_params = sum(p.numel() for p in engine.heads.parameters() if p.requires_grad)
    print(f"  Task Heads Trainable: {head_params:,}")

    # ====================================================
    # 5. TRAINING LOOP
    # ====================================================
    print("\n[4/5] Starting Training Loop...")
    
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\n{'='*30} EPOCH {epoch+1}/{EPOCHS} {'='*30}")
        
        # Train
        avg_train_loss = engine.train_one_epoch(optimizer)
        print(f"  >> Avg Train Loss: {avg_train_loss:.4f}")
        
        # Evaluate
        val_metrics = engine.evaluate(split_name="Validation")
        
        # Checkpointing Logic
        # We'll use a simple heuristic: if Cloning Acc + SSP Acc - Thermo MSE is good.
        # For simplicity here, we just save every epoch, or strictly track one metric.
        # Let's just save the latest model for now.
        save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save(engine.state_dict(), save_path)
        print(f"  >> Saved checkpoint: {save_path}")

    # ====================================================
    # 6. FINAL TEST
    # ====================================================
    print("\n[5/5] Final Test Set Evaluation...")
    engine.evaluate(loader_list=engine.test_loaders, split_name="TEST")
    print("\nDone!")

if __name__ == "__main__":
    main()