import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, matthews_corrcoef

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engines.engine_hf_with_task_specific_encoder import MultiTaskEngine 

SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_all_frozen" 
EPOCHS_TO_EVAL = [1, 2, 3, 4] 

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16 
LORA_RANK = 16
UNFROZEN_LAYERS = 0 

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
    print(f"  [Normalization] Mean: {mean:.4f} | Std: {std:.4f}")
    new_targets = []
    for t in all_raw_targets:
        if t is None: new_targets.append(None)
        else: new_targets.append((t - mean) / std)
    full_dataset.targets['target'] = new_targets

def get_detailed_metrics(engine, loader_list, split_name="TEST"):
    engine.eval()
    print(f"\n📊 Calculating Detailed Metrics for [{split_name}]...")
    results = {0: {'true': [], 'pred': []}, 1: {'true': [], 'pred': []}, 2: {'true': [], 'pred': []}}

    with torch.no_grad():
        for i, loader in enumerate(loader_list):
            task_type = engine.task_configs[i]['type']
            for batch in tqdm(loader, desc=f"Task {i}"):
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                is_token = (task_type in ['token_classification', 'per_residue_classification'])
                emb = engine.backbone(input_ids, mask, task_type='token' if is_token else 'sequence')
                logits = engine.heads[i](emb)
                
                if i == 0:
                    preds = logits.view(-1).cpu().numpy(); lbls = targets.view(-1).cpu().numpy()
                    results[0]['true'].extend(lbls); results[0]['pred'].extend(preds)
                elif i == 1:
                    preds = logits.argmax(dim=-1).cpu().numpy(); lbls = targets.cpu().numpy()
                    for b in range(lbls.shape[0]):
                        p_seq = preds[b]; t_seq = lbls[b]; valid_mask = (t_seq != -100)
                        results[1]['true'].extend(t_seq[valid_mask]); results[1]['pred'].extend(p_seq[valid_mask])
                elif i == 2:
                    preds = logits.argmax(dim=1).cpu().numpy(); lbls = targets.cpu().numpy()
                    results[2]['true'].extend(lbls); results[2]['pred'].extend(preds)

    print(f"\nResults for {split_name}:")
    t_mse = mean_squared_error(results[0]['true'], results[0]['pred'])
    t_r, _ = pearsonr(results[0]['true'], results[0]['pred'])
    print(f"  [Thermostability] MSE: {t_mse:.4f} | Pearson r: {t_r:.4f}")

    s_acc = accuracy_score(results[1]['true'], results[1]['pred'])
    s_f1 = f1_score(results[1]['true'], results[1]['pred'], average='macro')
    print(f"  [SecStructure]    Acc: {s_acc:.4f} | Macro F1:  {s_f1:.4f}")

    c_acc = accuracy_score(results[2]['true'], results[2]['pred'])
    c_mcc = matthews_corrcoef(results[2]['true'], results[2]['pred'])
    print(f"  [Cloning]         Acc: {c_acc:.4f} | MCC:       {c_mcc:.4f}")

def run_evaluation():
    set_seed(SEED)
    print(f"🚀 Starting Equal Weighting Evaluation on {DEVICE}...\n")
    
    print("[1] Loading Data...")
    ds_thermo = Thermostability(verbose=0); thermo_train, thermo_valid, thermo_test = ds_thermo.split()
    normalize_regression_targets(thermo_train, thermo_valid, thermo_test)
    ds_ssp = SecondaryStructure(verbose=0); ssp_train, ssp_valid, ssp_test = ds_ssp.split()
    ds_clf = CloningCLF(verbose=0); clf_train, clf_valid, clf_test = ds_clf.split()

    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    print("[2] Initializing Standard MultiTask Engine...")
    backbone = SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=UNFROZEN_LAYERS)
    # Using Standard Engine here
    engine = MultiTaskEngine(
        backbone=backbone, task_configs=task_configs,
        train_sets=[thermo_train], valid_sets=[thermo_valid, ssp_valid, clf_valid],
        test_sets=[thermo_test, ssp_test, clf_test],
        batch_size=BATCH_SIZE, device=DEVICE
    )

    for epoch in EPOCHS_TO_EVAL:
        filepath = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pt")
        if not os.path.exists(filepath): continue
        print(f"\n=== EVALUATING EPOCH {epoch} ===")
        engine.load_state_dict(torch.load(filepath, map_location=DEVICE), strict=False)
        get_detailed_metrics(engine, engine.valid_loaders, split_name="VALIDATION")
        get_detailed_metrics(engine, engine.test_loaders, split_name="TEST FINAL")

if __name__ == "__main__":
    run_evaluation()