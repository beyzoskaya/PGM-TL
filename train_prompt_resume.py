import os
import torch
import numpy as np
import csv
import random
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
LEARNING_RATE = 1e-4
LORA_RANK = 16
UNFROZEN_LAYERS = 2

START_EPOCH = 5       
TOTAL_EPOCHS = 8      
SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/framework_prompt_pcgrad"
PLOT_DIR = os.path.join(SAVE_DIR, "plots")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, f"model_epoch_{START_EPOCH}.pt")

#os.makedirs(PLOT_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_epoch_diagnostics(results, epoch):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Prompt-Tuning Diagnostics - Epoch {epoch}", fontsize=16)

    # 1. Thermostability Scatter
    ax = axes[0]
    t_true = results['Thermo']['true']
    t_pred = results['Thermo']['pred']
    sns.scatterplot(x=t_true, y=t_pred, ax=ax, alpha=0.3, color="purple")
    
    # Draw Ideal Line
    if len(t_true) > 0:
        low = min(min(t_true), min(t_pred))
        high = max(max(t_true), max(t_pred))
        ax.plot([low, high], [low, high], 'k--', label="Ideal")
    
    ax.set_title("Thermostability")
    ax.set_xlabel("True Z-Score")
    ax.set_ylabel("Pred Z-Score")

    # 2. SecStructure Confusion Matrix
    ax = axes[1]
    s_true = results['SSP']['true']
    s_pred = results['SSP']['pred']
    if len(s_true) > 0:
        cm = confusion_matrix(s_true, s_pred, normalize='true')
        sns.heatmap(cm, ax=ax, cmap="Blues", annot=False)
    ax.set_title("Secondary Structure Confusion")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    # 3. Cloning Histogram
    ax = axes[2]
    c_true = np.array(results['Cloning']['true'])
    c_prob = np.array(results['Cloning']['probs'])
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
    print(f"🔄 Preparing to Resume from Epoch {START_EPOCH} with Plots...")
    set_seed(SEED)

    print("[1/5] Re-loading Datasets (Deterministic)...")
    ds_t = Thermostability(verbose=0)
    t_train, t_val, t_test = ds_t.split()
    
    # Re-calculate Normalization
    full_dataset = t_train.dataset
    train_indices = t_train.indices
    all_raw = full_dataset.targets['target']
    
    vals = [all_raw[i] for i in train_indices if all_raw[i] is not None]
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    
    print(f"   - Thermo Train Size: {len(vals)}")
    print(f"   - Calculated Mean:   {mean_val:.4f}")
    print(f"   - Calculated Std:    {std_val:.4f}")
    
    new_t = []
    for t in all_raw:
        if t is None: new_t.append(None)
        else: new_t.append((t - mean_val) / std_val)
    full_dataset.targets['target'] = new_t

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

    print("[2/5] Re-building Model Architecture...")
    backbone = SharedProtBert(
        model_name="Rostlab/prot_bert_bfd",
        lora_rank=LORA_RANK,
        lora_dropout=0.1,
        unfrozen_layers=UNFROZEN_LAYERS 
    )

    engine = TaskPromptedEngine(
        backbone=backbone, task_configs=task_configs,
        train_sets=train_sets, valid_sets=valid_sets, test_sets=test_sets,
        batch_size=BATCH_SIZE, device=DEVICE, save_dir=SAVE_DIR
    )

    print(f"[3/5] Loading Weights from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found! Check path: {CHECKPOINT_PATH}")
    
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    engine.load_state_dict(state, strict=True)
    print("      ✓ Model weights successfully restored.")

    print("[4/5] Restoring Training State...")
    optimizer = torch.optim.AdamW(engine.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Fix for 'initial_lr' Key Error
    for group in optimizer.param_groups:
        group['initial_lr'] = LEARNING_RATE

    loader_lens = [len(l) for l in engine.train_loaders]
    steps_per_epoch = max(loader_lens)
    total_steps = steps_per_epoch * TOTAL_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    
    steps_completed = steps_per_epoch * START_EPOCH
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps,
        last_epoch=steps_completed - 1 
    )
    
    current_lr = scheduler.get_last_lr()[0]
    print(f"      - Resuming LR:     {current_lr:.2e}")

    print(f"\nRESUMING TRAINING (Epoch {START_EPOCH + 1} to {TOTAL_EPOCHS})...")
    
    history_path = os.path.join(SAVE_DIR, "training_history_resume.csv")
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ["Epoch", "Train_Loss"] 
        for t in task_configs: headers.append(f"Val_{t['name']}")
        for t in task_configs: headers.append(f"Test_{t['name']}")
        writer.writerow(headers)

    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        print(f"\n{'='*30} EPOCH {epoch+1}/{TOTAL_EPOCHS} {'='*30}")
        
        # Train
        train_res = engine.train_one_epoch(optimizer, scheduler, epoch_index=epoch+1)
        print(f"\n  [TRAIN] Avg Loss: {train_res['avg_loss']:.4f}")

        # Validate & Plot
        # We capture val_raw_data here
        val_metrics, val_raw_data = engine.evaluate(loader_list=engine.valid_loaders, split_name="VALIDATION")
        
        # Call Plotting Function
        plot_epoch_diagnostics(val_raw_data, epoch+1)

        # Test
        test_metrics, _ = engine.evaluate(loader_list=engine.test_loaders, split_name="TEST")

        # Log
        row = [epoch+1, train_res['avg_loss']]
        def parse_metric(m_dict, name):
            s = m_dict.get(name, "0.0")
            try: return float(s.split(":")[-1].strip())
            except: return 0.0

        for t in task_configs: row.append(parse_metric(val_metrics, t['name']))
        for t in task_configs: row.append(parse_metric(test_metrics, t['name']))

        with open(history_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        # Save Checkpoint
        save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save(engine.state_dict(), save_path)
        print(f"Checkpoint saved: {save_path}")

    print("\n✅ Resume Complete.")

if __name__ == "__main__":
    main()