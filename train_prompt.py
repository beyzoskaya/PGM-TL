# --- START OF FILE train_prompt.py ---

import os
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from transformers import get_linear_schedule_with_warmup

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_prompt import TaskPromptedEngine

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16        
EPOCHS = 8 
LEARNING_RATE = 1e-4

UNFROZEN_LAYERS = 2 
LORA_RANK = 16

SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/framework_prompt_early_fusion"
PLOT_DIR = os.path.join(SAVE_DIR, "plots")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

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
    print(f"  ✓ Normalization complete. Mean: {mean:.4f}, Std: {std:.4f}")

def plot_epoch_diagnostics(results, epoch):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Prompt-Tuning Diagnostics - Epoch {epoch}", fontsize=16)

    # 1. Thermostability Scatter
    ax = axes[0]
    t_true = results['Thermo']['true']; t_pred = results['Thermo']['pred']
    if len(t_true) > 0:
        sns.scatterplot(x=t_true, y=t_pred, ax=ax, alpha=0.3, color="purple")
        low, high = min(min(t_true), min(t_pred)), max(max(t_true), max(t_pred))
        ax.plot([low, high], [low, high], 'k--', label="Ideal")
        ax.set_title("Thermostability")
        ax.set_xlabel("True Z-Score"); ax.set_ylabel("Pred Z-Score")

    # 2. SecStructure Confusion Matrix
    ax = axes[1]
    s_true = results['SSP']['true']; s_pred = results['SSP']['pred']
    if len(s_true) > 0:
        # Use simple heatmap
        cm = confusion_matrix(s_true, s_pred, normalize='true')
        sns.heatmap(cm, ax=ax, cmap="Blues", annot=False)
        ax.set_title("Secondary Structure Confusion")
        ax.set_xlabel("Predicted Class"); ax.set_ylabel("True Class")

    # 3. Cloning Histogram
    ax = axes[2]
    c_true = np.array(results['Cloning']['true']); c_prob = np.array(results['Cloning']['probs'])
    if len(c_true) > 0:
        df_c = pd.DataFrame({"Prob": c_prob, "Truth": c_true})
        sns.histplot(data=df_c, x="Prob", hue="Truth", bins=20, ax=ax, palette={0: "red", 1: "green"}, kde=True, element="step")
        ax.set_title("Cloning Confidence")
        ax.set_xlabel("Predicted Probability (Soluble)")

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"epoch_{epoch}_diagnostics.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Diagnostic plot saved: {save_path}")

def main():
    set_seed(SEED)
    print(f"Running Prompt-Tuning + PCGrad on {DEVICE}")

    # 1. DATA LOADING
    print("[1/5] Loading Datasets...")
    
    # Thermostability
    ds_t = Thermostability(verbose=0)
    t_train, t_val, t_test = ds_t.split()
    normalize_regression_targets(t_train, t_val, t_test)
    
    # Secondary Structure
    ds_s = SecondaryStructure(verbose=0)
    s_train, s_val, s_test = ds_s.split()
    
    # Cloning
    ds_c = CloningCLF(verbose=0)
    c_train, c_val, c_test = ds_c.split()

    train_sets = [t_train, s_train, c_train]
    valid_sets = [t_val, s_val, c_val]
    test_sets  = [t_test, s_test, c_test]

    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    # 2. MODEL INITIALIZATION
    print("[2/5] Initializing Backbone...")
    backbone = SharedProtBert(
        model_name="Rostlab/prot_bert_bfd",
        lora_rank=LORA_RANK,
        lora_alpha=32,
        lora_dropout=0.1,
        unfrozen_layers=UNFROZEN_LAYERS 
    )

    # 3. ENGINE SETUP
    print("[3/5] Setting up Prompted Engine...")
    engine = TaskPromptedEngine(
        backbone=backbone, 
        task_configs=task_configs,
        train_sets=train_sets, 
        valid_sets=valid_sets, 
        test_sets=test_sets,
        batch_size=BATCH_SIZE, 
        device=DEVICE, 
        save_dir=SAVE_DIR
    )

    print(f"   [Check] Number of Prompts: {len(engine.task_prompts)}")

    # 4. OPTIMIZATION
    # We optimize both the backbone (LoRA) and the Engine (Prompts + Heads)
    optimizer = torch.optim.AdamW(engine.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Calculate total steps for scheduler
    max_len = max([len(l) for l in engine.train_loaders])
    total_steps = max_len * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    # 5. LOGGING INIT
    history_path = os.path.join(SAVE_DIR, "training_history.csv")
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ["Epoch", "Train_Combined_Loss"] 
        for t in task_configs: headers.append(f"Val_{t['name']}")
        for t in task_configs: headers.append(f"Test_{t['name']}")
        writer.writerow(headers)

    best_val_thermo_loss = float('inf')

    # 6. TRAINING LOOP
    for epoch in range(EPOCHS):
        print(f"\n{'='*30} EPOCH {epoch+1}/{EPOCHS} {'='*30}")
        
        # --- TRAIN ---
        # Note: We do NOT pass debug=True here, so logs stay clean
        train_res = engine.train_one_epoch(optimizer, scheduler, epoch+1)
        print(f"\n  [TRAIN] Combined Loss: {train_res['avg_loss']:.4f}")

        # --- VALIDATE ---
        val_metrics, val_raw_data = engine.evaluate(loader_list=engine.valid_loaders, split_name="VALIDATION")
        plot_epoch_diagnostics(val_raw_data, epoch+1)

        # --- TEST ---
        test_metrics, _ = engine.evaluate(loader_list=engine.test_loaders, split_name="TEST")

        # --- SAVE LOGS ---
        row = [epoch+1, train_res['avg_loss']]
        
        # Parse Val Metrics
        for t in task_configs:
            val_str = val_metrics.get(t['name'], "0.0")
            try: row.append(float(val_str.split(":")[-1].strip()))
            except: row.append(0.0)
            
        # Parse Test Metrics
        for t in task_configs:
            test_str = test_metrics.get(t['name'], "0.0")
            try: row.append(float(test_str.split(":")[-1].strip()))
            except: row.append(0.0)

        with open(history_path, 'a', newline='') as f: csv.writer(f).writerow(row)

        # --- CHECKPOINTING ---
        torch.save(engine.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt"))
        
        # Save Best Model based on Thermostability (Regression is usually the hardest to stabilize)
        try:
            curr_val_mse = float(val_metrics['Thermostability'].split(":")[1])
            if curr_val_mse < best_val_thermo_loss:
                best_val_thermo_loss = curr_val_mse
                torch.save(engine.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
                print(f"  🌟 New Best Prompt Model Saved! (Val Thermo MSE: {best_val_thermo_loss:.4f})")
        except: 
            pass

    print("\nDone!")

if __name__ == "__main__":
    main()