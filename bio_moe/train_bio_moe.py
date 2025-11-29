import os
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import get_linear_schedule_with_warmup

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from bio_moe.engine_hf_bio_moe import BioMoE_Engine

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16        
EPOCHS = 5 
LEARNING_RATE = 1e-4

SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/bio_moe_v1"
PLOT_DIR = os.path.join(SAVE_DIR, "plots")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

LORA_RANK = 16
UNFROZEN_LAYERS = 0 

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalize_regression_targets(train_ds, valid_ds, test_ds):
    print("\n[Data] Normalizing Thermostability targets...")
    full_dataset = train_ds.dataset; train_indices = train_ds.indices
    all_raw = full_dataset.targets['target']
    vals = [all_raw[i] for i in train_indices if all_raw[i] is not None]
    mean = np.mean(vals); std = np.std(vals)
    new_t = []
    for t in all_raw:
        if t is None: new_t.append(None)
        else: new_t.append((t-mean)/std)
    full_dataset.targets['target'] = new_t
    print("  ✓ Normalization complete.")

def plot_epoch_diagnostics(results, epoch):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Bio-MoE Diagnostics - Epoch {epoch}", fontsize=20)

    # 1. Heatmap
    avg_weights = []
    task_names = ["Thermo", "Structure", "Cloning"]
    for key in ['Thermo', 'SSP', 'Cloning']:
        w_list = results[key]['weights']
        if w_list:
            w_stack = np.stack(w_list)
            avg_weights.append(np.mean(w_stack, axis=0))
        else:
            avg_weights.append([0,0,0])
    heatmap_data = np.stack(avg_weights)
    sns.heatmap(heatmap_data, ax=axes[0, 0], annot=True, cmap="YlGnBu", 
                xticklabels=["Exp: Thermo", "Exp: Struct", "Exp: Clone"], 
                yticklabels=task_names, vmin=0, vmax=1)
    axes[0, 0].set_title("Router Attention (Task vs Expert)")

    # 2. Thermo Scatter
    t_true = results['Thermo']['true']; t_pred = results['Thermo']['pred']
    sns.scatterplot(x=t_true, y=t_pred, ax=axes[0, 1], alpha=0.3, color="purple")
    low, high = min(min(t_true), min(t_pred)), max(max(t_true), max(t_pred))
    axes[0, 1].plot([low, high], [low, high], 'k--', label="Ideal")
    axes[0, 1].set_title("Thermostability (Validation)")

    # 3. Cloning Hist
    c_true = np.array(results['Cloning']['true']); c_prob = np.array(results['Cloning']['prob'])
    df_c = pd.DataFrame({"Prob": c_prob, "Truth": c_true})
    sns.histplot(data=df_c, x="Prob", hue="Truth", bins=20, ax=axes[1, 0], palette={0: "red", 1: "green"}, kde=True, element="step")
    axes[1, 0].set_title("Cloning Solubility Confidence")

    # 4. Stats
    axes[1, 1].axis('off')
    txt = "Router Analysis:\n\n"
    for i, name in enumerate(task_names):
        w = avg_weights[i]
        txt += f"Task {name} relies on:\n  - Thermo: {w[0]:.2f}\n  - Struct: {w[1]:.2f}\n  - Clone:  {w[2]:.2f}\n\n"
    axes[1, 1].text(0.1, 0.5, txt, fontsize=12, va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"epoch_{epoch}_diagnostics.png"))
    plt.close()

def main():
    set_seed(SEED)
    print(f"🚀 Running BIO-MOE (LoRA + Fusion Enabled) on {DEVICE}")

    print("[1/5] Loading Datasets...")
    ds_t = Thermostability(verbose=0); t_train, t_val, t_test = ds_t.split()
    normalize_regression_targets(t_train, t_val, t_test)
    ds_s = SecondaryStructure(verbose=0); s_train, s_val, s_test = ds_s.split()
    ds_c = CloningCLF(verbose=0); c_train, c_val, c_test = ds_c.split()

    train_sets = [t_train, s_train, c_train]
    valid_sets = [t_val, s_val, c_val]
    test_sets  = [t_test, s_test, c_test]

    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    print("[2/5] Initializing Backbone...")
    # NOTE: This initializes with LoRA enabled and trainable!
    backbone = SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=UNFROZEN_LAYERS)

    print("[3/5] Setting up Bio-MoE Engine...")
    engine = BioMoE_Engine(
        backbone=backbone, task_configs=task_configs,
        train_sets=train_sets, valid_sets=valid_sets, test_sets=test_sets,
        batch_size=BATCH_SIZE, device=DEVICE, save_dir=SAVE_DIR 
    )

    # CHECK PARAMETERS
    trainable = sum(p.numel() for p in engine.parameters() if p.requires_grad)
    print(f"\n🔥 Total Trainable Params (LoRA + Experts): {trainable:,}")
    print(f"   (Should be around 11 Million)\n")

    optimizer = torch.optim.AdamW(engine.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    max_len = max([len(l) for l in engine.train_loaders])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*max_len*EPOCHS), num_training_steps=max_len*EPOCHS)

    history_path = os.path.join(SAVE_DIR, "training_history.csv")
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ["Epoch", "Train_Combined_Loss"] + [f"Val_{t['name']}" for t in task_configs]
        writer.writerow(headers)

    best_val_thermo_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\n{'='*30} EPOCH {epoch+1}/{EPOCHS} {'='*30}")
        
        # Pass scheduler to train_one_epoch
        train_res = engine.train_one_epoch(optimizer, scheduler, epoch+1)
        print(f"\n  [TRAIN] Combined Loss: {train_res['avg_loss']:.4f}")

        metrics, raw_data = engine.evaluate(loader_list=engine.valid_loaders, split_name="VALIDATION")
        plot_epoch_diagnostics(raw_data, epoch+1)

        row = [epoch+1, train_res['avg_loss']]
        for t in task_configs:
            val_str = metrics.get(t['name'], "0.0")
            try: row.append(float(val_str.split(":")[-1].strip()))
            except: row.append(0.0)
        with open(history_path, 'a', newline='') as f: csv.writer(f).writerow(row)

        torch.save(engine.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt"))
        
        try:
            curr_mse = float(metrics['Thermostability'].split(":")[1])
            if curr_mse < best_val_thermo_loss:
                best_val_thermo_loss = curr_mse
                torch.save(engine.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
                print(f" New Best Bio-MoE Saved! (Thermo MSE: {best_val_thermo_loss:.4f})")
        except: pass

    print("\nDone!")

if __name__ == "__main__":
    main()