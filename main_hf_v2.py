import os
import torch
import numpy as np
import logging
import csv
import json

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert

from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid

# --- CONFIGURATION ---
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16        
EPOCHS = 5 # Keep 5 to compare directly with the failed run
LEARNING_RATE = 1e-4

SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad"
os.makedirs(SAVE_DIR, exist_ok=True)

LORA_RANK = 16
UNFROZEN_LAYERS = 0 # Frozen + LoRA as the baseline

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_regression_targets(train_ds, valid_ds, test_ds):
    print("\n[Data] Normalizing Thermostability targets...")
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
    print(f"Running HYBRID (PCGrad + Uncertainty) on {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS}")

    # 1. DATA LOADING
    print("\n[1/5] Loading Datasets...")
    ds_thermo = Thermostability(verbose=0) 
    thermo_train, thermo_valid, thermo_test = ds_thermo.split()
    normalize_regression_targets(thermo_train, thermo_valid, thermo_test)

    ds_ssp = SecondaryStructure(verbose=0)
    ssp_train, ssp_valid, ssp_test = ds_ssp.split()

    ds_clf = CloningCLF(verbose=0)
    clf_train, clf_valid, clf_test = ds_clf.split()

    print(f"  Thermo Train: {len(thermo_train)} | SSP Train: {len(ssp_train)} | Cloning Train: {len(clf_train)}")

    train_sets = [thermo_train, ssp_train, clf_train]
    valid_sets = [thermo_valid, ssp_valid, clf_valid]
    test_sets  = [thermo_test,  ssp_test,  clf_test]

    # 2. CONFIGURATION
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    # 3. MODEL INITIALIZATION
    print(f"\n[2/5] Initializing SharedProtBert (LoRA + Top {UNFROZEN_LAYERS} Unfrozen)...")
    backbone = SharedProtBert(
        model_name="Rostlab/prot_bert_bfd",
        lora_rank=LORA_RANK,
        lora_dropout=0.1,
        unfrozen_layers=UNFROZEN_LAYERS
    )

    # 4. ENGINE SETUP
    print("\n[3/5] Setting up HYBRID Engine (PCGrad + Uncertainty)...")
    
    engine = MultiTaskEngineHybrid(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=train_sets,
        valid_sets=valid_sets,
        test_sets=test_sets,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        save_dir=SAVE_DIR 
    )

    optimizer = torch.optim.AdamW(engine.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # --- SETUP LOGGING CSV ---
    history_path = os.path.join(SAVE_DIR, "training_history.csv")
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ["Epoch", "Train_Combined_Loss"]
        for task in task_configs:
            name = task['name']
            headers.extend([f"Train_Loss_{name}", f"Val_Loss_{name}"])
        writer.writerow(headers)

    # 5. TRAINING LOOP
    print("\n[4/5] Starting Training Loop...")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*30} EPOCH {epoch+1}/{EPOCHS} {'='*30}")
        
        # A. TRAIN
        train_metrics = engine.train_one_epoch(optimizer, epoch_index=epoch+1)
        
        # Print Train Results
        print(f"\n  [TRAIN]")
        print(f"  >> Combined (Hybrid) Loss: {train_metrics.pop('combined_loss'):.4f}")
        for t, m in train_metrics.items(): print(f"  >> {t}: {m}")

        # B. VALIDATE
        val_metrics = engine.evaluate(loader_list=engine.valid_loaders, split_name="VALIDATION")

        # C. TEST
        engine.evaluate(loader_list=engine.test_loaders, split_name="TEST")

        # D. LOGGING & SAVING
        row = [epoch+1, 0.0] 
        current_val_combined_raw = 0
        
        for task in task_configs:
            t_name = task['name']
            row.append(0) 
            
            v_loss = val_metrics.get(t_name, {}).get('Loss', val_metrics.get(t_name, {}).get('MSE', 0))
            row.append(v_loss)
            current_val_combined_raw += v_loss
        
        with open(history_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save(engine.state_dict(), save_path)
        print(f"  >> Saved: {save_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()