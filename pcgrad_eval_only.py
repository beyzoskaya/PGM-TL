import os
import torch
import numpy as np
import csv
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, matthews_corrcoef

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_with_uncertanity_weighting import MultiTaskEngineUncertanityWeighting
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16 
LORA_RANK = 16
UNFROZEN_LAYERS = 0 

SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad"
EPOCHS_TO_EVAL = [1,2,3,4,5] 

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalize_regression_targets(train_ds, valid_ds, test_ds):
    """
    Calculates Mean/Std on TRAIN set only, then applies to Valid/Test.
    This prevents data leakage.
    """
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
    return mean, std

def get_detailed_metrics(engine, loader_list, split_name="TEST"):
    """
    Runs inference and calculates advanced metrics:
    - Thermo: MSE, Pearson
    - SSP: Accuracy, Macro-F1
    - Cloning: Accuracy, MCC
    """
    engine.eval()
    print(f"\n📊 Calculating Detailed Metrics for [{split_name}]...")
    
    # Containers for results
    results = {
        0: {'true': [], 'pred': []}, # Thermo
        1: {'true': [], 'pred': []}, # SSP
        2: {'true': [], 'pred': []}  # Cloning
    }

    with torch.no_grad():
        # Iterate over tasks
        for i, loader in enumerate(loader_list):
            task_type = engine.task_configs[i]['type']
            
            for batch in tqdm(loader, desc=f"Task {i}"):
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                
                is_token = (task_type in ['token_classification', 'per_residue_classification'])
                
                # Forward Pass
                emb = engine.backbone(input_ids, mask, task_type='token' if is_token else 'sequence')
                logits = engine.heads[i](emb)
                
                # Collect Data
                if i == 0: # Thermostability (Regression)
                    preds = logits.view(-1).cpu().numpy()
                    lbls = targets.view(-1).cpu().numpy()
                    results[0]['true'].extend(lbls)
                    results[0]['pred'].extend(preds)
                    
                elif i == 1: # SSP (Token Classif)
                    preds = logits.argmax(dim=-1).cpu().numpy() # [B, L]
                    lbls = targets.cpu().numpy()                # [B, L]
                    
                    # Flatten and remove padding (-100)
                    for b in range(lbls.shape[0]):
                        p_seq = preds[b]
                        t_seq = lbls[b]
                        valid_mask = (t_seq != -100)
                        results[1]['true'].extend(t_seq[valid_mask])
                        results[1]['pred'].extend(p_seq[valid_mask])
                        
                elif i == 2: # Cloning (Binary Classif)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    lbls = targets.cpu().numpy()
                    results[2]['true'].extend(lbls)
                    results[2]['pred'].extend(preds)

    # --- CALCULATE METRICS ---
    print(f"\nresults for {split_name}:")
    
    # 1. Thermostability
    t_true = results[0]['true']
    t_pred = results[0]['pred']
    mse = mean_squared_error(t_true, t_pred)
    pearson, _ = pearsonr(t_true, t_pred)
    print(f"  [Thermostability]")
    print(f"    MSE: {mse:.4f}")
    print(f"    Pearson r: {pearson:.4f}  <-- Check this!")

    # 2. Secondary Structure
    s_true = results[1]['true']
    s_pred = results[1]['pred']
    s_acc = accuracy_score(s_true, s_pred)
    s_f1 = f1_score(s_true, s_pred, average='macro')
    print(f"  [SecStructure]")
    print(f"    Accuracy: {s_acc:.4f}")
    print(f"    Macro F1: {s_f1:.4f}      <-- Better for imbalanced classes")

    # 3. Cloning
    c_true = results[2]['true']
    c_pred = results[2]['pred']
    c_acc = accuracy_score(c_true, c_pred)
    c_mcc = matthews_corrcoef(c_true, c_pred)
    print(f"  [Cloning]")
    print(f"    Accuracy: {c_acc:.4f}")
    print(f"    MCC:      {c_mcc:.4f}     <-- Gold standard for binary")

def run_evaluation():
    set_seed(SEED)
    print(f"🚀 Starting PCGrad Metric Evaluation on {DEVICE}...\n")

    # 1. Load Data
    print("[1/4] Loading and Splitting Datasets...")
    ds_thermo = Thermostability(verbose=0); thermo_train, thermo_valid, thermo_test = ds_thermo.split()
    # IMPORTANT: Calculate normalization on TRAIN, apply to VALID/TEST
    normalize_regression_targets(thermo_train, thermo_valid, thermo_test)
    
    ds_ssp = SecondaryStructure(verbose=0); ssp_train, ssp_valid, ssp_test = ds_ssp.split()
    ds_clf = CloningCLF(verbose=0); clf_train, clf_valid, clf_test = ds_clf.split()

    # We need Valid and Test sets
    valid_sets = [thermo_valid, ssp_valid, clf_valid]
    test_sets  = [thermo_test,  ssp_test,  clf_test]

    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    # 2. Initialize Model
    print("[2/4] Initializing SharedProtBert...")
    backbone = SharedProtBert(
        model_name="Rostlab/prot_bert_bfd",
        lora_rank=LORA_RANK,
        lora_dropout=0.1,
        unfrozen_layers=UNFROZEN_LAYERS
    )

    # Use HybridEngine (PCGrad)
    engine = MultiTaskEngineHybrid(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=[thermo_train], # Dummy, not used for eval
        valid_sets=valid_sets,
        test_sets=test_sets,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        save_dir=SAVE_DIR
    )

    # 3. Loop over Epochs
    for epoch in EPOCHS_TO_EVAL:
        filename = f"model_epoch_{epoch}.pt"
        filepath = os.path.join(SAVE_DIR, filename)
        
        print(f"\n{'='*40}")
        print(f"EVALUATING EPOCH {epoch}")
        print(f"{'='*40}")
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue
        
        print(f"Loading weights: {filename} ...")
        # strict=False allows loading even if internal optimizer buffers are missing
        state_dict = torch.load(filepath, map_location=DEVICE)
        engine.load_state_dict(state_dict, strict=False)
        
        # 4. Run Custom Metric Calculation
        # Validation Set
        get_detailed_metrics(engine, engine.valid_loaders, split_name="VALIDATION")
        
        # Test Set
        get_detailed_metrics(engine, engine.test_loaders, split_name="TEST FINAL")

if __name__ == "__main__":
    run_evaluation()