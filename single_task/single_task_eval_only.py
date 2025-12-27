import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, matthews_corrcoef

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engines.engine_hf_with_task_specific_encoder import MultiTaskEngine

SINGLE_TASK_PATHS = {
    'Thermostability': "/content/drive/MyDrive/protein_multitask_outputs/baseline_Thermostability_lora_only",
    'SecStructure':    "/content/drive/MyDrive/protein_multitask_outputs/baseline_SecStructure_lora_only",
    'Cloning':         "/content/drive/MyDrive/protein_multitask_outputs/baseline_Cloning_lora_only"
}
EPOCHS_TO_EVAL = [1, 2, 3, 4, 5] 

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

def run_single_task_evaluation():
    set_seed(SEED)
    print(f"🚀 Starting Single Task Baseline Evaluation on {DEVICE}...\n")

    # 1. Load All Data First
    print("[1] Loading Data...")
    ds_thermo = Thermostability(verbose=0); t_train, t_valid, t_test = ds_thermo.split()
    normalize_regression_targets(t_train, t_valid, t_test)
    ds_ssp = SecondaryStructure(verbose=0); s_train, s_valid, s_test = ds_ssp.split()
    ds_clf = CloningCLF(verbose=0); c_train, c_valid, c_test = ds_clf.split()

    # Definitions for looping
    tasks = [
        ('Thermostability', {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1}, t_valid, t_test),
        ('SecStructure',    {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8}, s_valid, s_test),
        ('Cloning',         {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}, c_valid, c_test)
    ]

    for task_name, config, valid_set, test_set in tasks:
        print(f"\n\n{'#'*40}")
        print(f"EVALUATING BASELINE: {task_name}")
        print(f"{'#'*40}")

        model_dir = SINGLE_TASK_PATHS[task_name]
        
        # Initialize Engine for just this one task
        backbone = SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=UNFROZEN_LAYERS)
        engine = MultiTaskEngine(
            backbone=backbone, task_configs=[config],
            train_sets=[valid_set], # Dummy
            valid_sets=[valid_set], test_sets=[test_set],
            batch_size=BATCH_SIZE, device=DEVICE
        )

        for epoch in EPOCHS_TO_EVAL:
            filepath = os.path.join(model_dir, f"model_epoch_{epoch}.pt")
            if not os.path.exists(filepath): continue
            
            print(f"\n--- Epoch {epoch} ---")
            engine.load_state_dict(torch.load(filepath, map_location=DEVICE), strict=False)
            engine.eval()
            
            # Helper to run inference
            def calc_metric(loader, split):
                all_true, all_pred = [], []
                with torch.no_grad():
                    for batch in loader:
                        input_ids = batch['input_ids'].to(DEVICE)
                        mask = batch['attention_mask'].to(DEVICE)
                        targets = batch['targets'].to(DEVICE)
                        
                        is_token = (config['type'] == 'token_classification')
                        emb = engine.backbone(input_ids, mask, task_type='token' if is_token else 'sequence')
                        logits = engine.heads[0](emb) # Only 1 head exists
                        
                        if task_name == 'Thermostability':
                            all_pred.extend(logits.view(-1).cpu().numpy())
                            all_true.extend(targets.view(-1).cpu().numpy())
                        elif task_name == 'SecStructure':
                            preds = logits.argmax(dim=-1).cpu().numpy(); lbls = targets.cpu().numpy()
                            for b in range(lbls.shape[0]):
                                valid = (lbls[b] != -100)
                                all_true.extend(lbls[b][valid]); all_pred.extend(preds[b][valid])
                        elif task_name == 'Cloning':
                            all_pred.extend(logits.argmax(dim=1).cpu().numpy())
                            all_true.extend(targets.cpu().numpy())
                
                print(f"[{split}]")
                if task_name == 'Thermostability':
                    mse = mean_squared_error(all_true, all_pred)
                    r, _ = pearsonr(all_true, all_pred)
                    print(f"  MSE: {mse:.4f} | Pearson: {r:.4f}")
                elif task_name == 'SecStructure':
                    acc = accuracy_score(all_true, all_pred)
                    f1 = f1_score(all_true, all_pred, average='macro')
                    print(f"  Acc: {acc:.4f} | Macro F1: {f1:.4f}")
                elif task_name == 'Cloning':
                    acc = accuracy_score(all_true, all_pred)
                    mcc = matthews_corrcoef(all_true, all_pred)
                    print(f"  Acc: {acc:.4f} | MCC: {mcc:.4f}")

            calc_metric(engine.valid_loaders[0], "VALIDATION")
            calc_metric(engine.test_loaders[0], "TEST FINAL")

if __name__ == "__main__":
    run_single_task_evaluation()