import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, matthews_corrcoef

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engines.engine_hf_cascade import CascadeMultiTaskEngine

SEED = 42
SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/cascade_v1_lora16"
EPOCHS_TO_EVAL = [1, 2, 3, 4, 5] 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16 
LORA_RANK = 16

UNFROZEN_LAYERS = 0 
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalize_regression_targets(train_ds):
    full_dataset = train_ds.dataset; train_indices = train_ds.indices
    all_raw_targets = full_dataset.targets['target']
    train_values = [all_raw_targets[i] for i in train_indices if all_raw_targets[i] is not None]
    mean = np.mean(train_values); std = np.std(train_values)
    new_targets = []
    for t in all_raw_targets:
        if t is None: new_targets.append(None)
        else: new_targets.append((t - mean) / std)
    full_dataset.targets['target'] = new_targets

def get_detailed_metrics(engine, loader_list, epoch_num):
    engine.eval()
    results = {0: {'true': [], 'pred': []}, 1: {'true': [], 'pred': []}, 2: {'true': [], 'pred': []}}

    with torch.no_grad():
        for i, loader in enumerate(loader_list):
            for batch in tqdm(loader, desc=f"Ep {epoch_num} - Task {i}", leave=False):
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                
                # Cascade Forward Pass
                ssp_p, thermo_p, clone_p = engine.forward(input_ids, mask)
                
                if i == 0: # Thermo
                    results[0]['true'].extend(targets.view(-1).cpu().numpy())
                    results[0]['pred'].extend(thermo_p.view(-1).cpu().numpy())
                elif i == 1: # SSP
                    preds = ssp_p.argmax(dim=-1).cpu().numpy(); lbls = targets.cpu().numpy()
                    for b in range(lbls.shape[0]):
                        valid = (lbls[b] != -100)
                        results[1]['true'].extend(lbls[b][valid]); results[1]['pred'].extend(preds[b][valid])
                elif i == 2: # Cloning
                    results[2]['true'].extend(targets.cpu().numpy())
                    results[2]['pred'].extend(clone_p.argmax(dim=1).cpu().numpy())

    print(f"\n📊 Results for Cascade Epoch {epoch_num}:")
    
    # 1. Thermostability
    t_r, _ = pearsonr(results[0]['true'], results[0]['pred'])
    print(f"  [Thermostability] Pearson r: {t_r:.4f}")

    # 2. SecStructure
    s_f1 = f1_score(results[1]['true'], results[1]['pred'], average='macro')
    print(f"  [SecStructure]    Macro F1:  {s_f1:.4f}")

    # 3. Cloning
    c_mcc = matthews_corrcoef(results[2]['true'], results[2]['pred'])
    print(f"  [Cloning]         MCC:       {c_mcc:.4f}")
    
    print("-" * 30)

def run_evaluation():
    set_seed(SEED)
    print(f"🚀 Eval: Full Cascade Metrics Scan on {DEVICE}...")
    
    ds_t = Thermostability(verbose=0); t_train, _, t_test = ds_t.split()
    normalize_regression_targets(t_train)
    ds_s = SecondaryStructure(verbose=0); _, _, s_test = ds_s.split()
    ds_c = CloningCLF(verbose=0); _, _, c_test = ds_c.split()

    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    backbone = SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=0)
    engine = CascadeMultiTaskEngine(
        backbone=backbone, task_configs=task_configs,
        train_sets=[t_train], valid_sets=[t_test], # Dummy
        test_sets=[t_test, s_test, c_test],
        batch_size=BATCH_SIZE, device=DEVICE, save_dir=SAVE_DIR
    )

    for epoch in EPOCHS_TO_EVAL:
        path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pt")
        if os.path.exists(path):
            engine.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
            get_detailed_metrics(engine, engine.valid_loaders, epoch)
            #get_detailed_metrics(engine, engine.test_loaders, epoch)
        else:
            print(f"Skipping Epoch {epoch} (File not found)")

if __name__ == "__main__":
    run_evaluation()