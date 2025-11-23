import os
import torch
import numpy as np
import csv

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_with_uncertanity_weighting import MultiTaskEngineUncertanityWeighting

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16        
EPOCHS = 5
LEARNING_RATE = 1e-4
# Use the EXACT same directory
SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_uncertainty"
LORA_RANK = 16
UNFROZEN_LAYERS = 0 

START_EPOCH_INDEX = 2

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalize_regression_targets(train_ds, valid_ds, test_ds):
    full_dataset = train_ds.dataset 
    train_indices = train_ds.indices
    all_raw_targets = full_dataset.targets['target']
    train_values = [all_raw_targets[i] for i in train_indices if all_raw_targets[i] is not None]
    mean = np.mean(train_values); std = np.std(train_values)
    new_targets = []
    for t in all_raw_targets:
        if t is None: new_targets.append(None)
        else: new_targets.append((t - mean) / std)
    full_dataset.targets['target'] = new_targets

def resume_training():
    set_seed(SEED)
    print(f"RESUMING TRAINING from Epoch {START_EPOCH_INDEX + 1}...")

    # 1. DATA
    print("[1/5] Loading Datasets...")
    ds_thermo = Thermostability(verbose=0); thermo_train, thermo_valid, thermo_test = ds_thermo.split()
    normalize_regression_targets(thermo_train, thermo_valid, thermo_test)
    ds_ssp = SecondaryStructure(verbose=0); ssp_train, ssp_valid, ssp_test = ds_ssp.split()
    ds_clf = CloningCLF(verbose=0); clf_train, clf_valid, clf_test = ds_clf.split()

    train_sets = [thermo_train, ssp_train, clf_train]
    valid_sets = [thermo_valid, ssp_valid, clf_valid]
    test_sets  = [thermo_test,  ssp_test,  clf_test]

    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    # 2. MODEL
    print(f"\n[2/5] Initializing Backbone...")
    backbone = SharedProtBert(
        model_name="Rostlab/prot_bert_bfd",
        lora_rank=LORA_RANK,
        lora_dropout=0.1,
        unfrozen_layers=UNFROZEN_LAYERS
    )

    # 3. ENGINE (Loads logs safely)
    print("\n[3/5] Setting up Engine...")
    engine = MultiTaskEngineUncertanityWeighting(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=train_sets,
        valid_sets=valid_sets,
        test_sets=test_sets,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        save_dir=SAVE_DIR
    )

    # 4. LOAD WEIGHTS
    checkpoint_path = os.path.join(SAVE_DIR, "model_epoch_1.pt")
    if os.path.exists(checkpoint_path):
        print(f"\n[4/5] Loading Checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        engine.load_state_dict(state_dict)
        print("   ✅ Weights loaded successfully.")
    else:
        print(f"\n❌ ERROR: {checkpoint_path} not found! Cannot resume.")
        return

    optimizer = torch.optim.AdamW(engine.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # 5. RESUME LOOP
    print(f"\n[5/5] Resuming Loop from Epoch {START_EPOCH_INDEX + 1} to {EPOCHS}...")
    history_path = os.path.join(SAVE_DIR, "training_history.csv")

    for epoch in range(START_EPOCH_INDEX, EPOCHS):
        print(f"\n{'='*30} EPOCH {epoch+1}/{EPOCHS} {'='*30}")
        
        # Train
        train_metrics = engine.train_one_epoch(optimizer, epoch_index=epoch+1)
        
        print(f"\n  [TRAIN]")
        print(f"  >> Combined (Weighted) Loss: {train_metrics.pop('combined_loss'):.4f}")
        for t, m in train_metrics.items(): print(f"  >> {t}: {m}")

        # Validate
        val_metrics = engine.evaluate(loader_list=engine.valid_loaders, split_name="VALIDATION")

        engine.evaluate(loader_list=engine.test_loaders, split_name="TEST")
        # -------------------------------------

        row = [epoch+1, 0.0] 
        for task in task_configs:
            t_name = task['name']
            row.append(0) 
            v_loss = val_metrics.get(t_name, {}).get('Loss', val_metrics.get(t_name, {}).get('MSE', 0))
            row.append(v_loss)
        
        with open(history_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        # Save
        save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save(engine.state_dict(), save_path)
        print(f"  >> Saved: {save_path}")

    print("\n✅ Training Complete!")

if __name__ == "__main__":
    resume_training()