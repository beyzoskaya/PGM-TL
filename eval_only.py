import os
import torch
import numpy as np

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_with_uncertanity_weighting import MultiTaskEngineUncertanityWeighting

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16 
LORA_RANK = 16
UNFROZEN_LAYERS = 0 

SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_uncertainty"

EPOCHS_TO_EVAL = [1, 2] 

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalize_regression_targets(train_ds, valid_ds, test_ds):
    full_dataset = train_ds.dataset 
    train_indices = train_ds.indices
    all_raw_targets = full_dataset.targets['target']
    
    train_values = [all_raw_targets[i] for i in train_indices if all_raw_targets[i] is not None]
    
    mean = np.mean(train_values)
    std = np.std(train_values)
    print(f"  [Normalization] Mean: {mean:.4f} | Std: {std:.4f}")

    new_targets = []
    for t in all_raw_targets:
        if t is None: new_targets.append(None)
        else: new_targets.append((t - mean) / std)
            
    full_dataset.targets['target'] = new_targets

def run_evaluation():
    set_seed(SEED)
    print(f"🚀 Starting Evaluation Mode on {DEVICE}...\n")

    print("[1/4] Loading and Splitting Datasets...")
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

    print("[2/4] Initializing Model Architecture (Blank)...")
    backbone = SharedProtBert(
        model_name="Rostlab/prot_bert_bfd",
        lora_rank=LORA_RANK,
        lora_dropout=0.1,
        unfrozen_layers=UNFROZEN_LAYERS
    )

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

    print(f"\n[3/4] Evaluating Checkpoints: {EPOCHS_TO_EVAL}")

    for epoch in EPOCHS_TO_EVAL:
        filename = f"model_epoch_{epoch}.pt"
        filepath = os.path.join(SAVE_DIR, filename)
        
        print(f"\n{'-'*20} EPOCH {epoch} {'-'*20}")
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue
        
        print(f"Loading weights: {filename} ...")
        state_dict = torch.load(filepath, map_location=DEVICE)
        
        try:
            engine.load_state_dict(state_dict)
            print("✓ Weights loaded successfully.")
        except RuntimeError as e:
            print(f"⚠️ Key Mismatch (Did you change architecture?): {e}")
            continue

        print("\n--- VALIDATION SET ---")
        engine.evaluate(loader_list=engine.valid_loaders, split_name="VALIDATION")
        
        print("\n--- TEST SET ---")
        engine.evaluate(loader_list=engine.test_loaders, split_name="TEST")

    print("\n✅ Evaluation Complete.")

if __name__ == "__main__":
    run_evaluation()