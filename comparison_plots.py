import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

from engine_hf_with_task_specific_encoder import MultiTaskEngine as StandardEngine
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid as HybridEngine
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
LORA_RANK = 16
UNFROZEN_LAYERS = 0

BASE_DIR = "/content/drive/MyDrive/protein_multitask_outputs"
PATH_UW_DIR = os.path.join(BASE_DIR, "cyclic_v1_lora16_uncertainty")
PATH_PC_DIR = os.path.join(BASE_DIR, "cyclic_v1_lora16_hybrid_pcgrad")

PATH_BASE_THERMO = os.path.join(BASE_DIR, "baseline_Thermostability_lora_only/model_epoch_5.pt") 
PATH_BASE_CLONING = os.path.join(BASE_DIR, "baseline_Cloning_lora_only/model_epoch_5.pt")      

# --- LOADERS ---

def get_backbone():
    return SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=UNFROZEN_LAYERS)

def load_baseline(task_name, checkpoint_path):
    """Loads a Single-Task model using the Standard Engine"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Baseline not found: {checkpoint_path}")
        return None
        
    backbone = get_backbone()
    
    # Reconstruct the config used during single-task training
    if task_name == 'Thermostability':
        cfg = [{'name': 'Thermostability', 'type': 'regression', 'num_labels': 1}]
    elif task_name == 'Cloning':
        cfg = [{'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}]
        
    # Init Standard Engine (List of 1 task)
    engine = StandardEngine(backbone, cfg, [], [], batch_size=BATCH_SIZE, device=DEVICE)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        engine.load_state_dict(state_dict)
        engine.eval()
        engine.to(DEVICE)
        print(f"✅ Loaded Baseline: {task_name}")
        return engine
    except Exception as e:
        print(f"❌ Error loading Baseline {task_name}: {e}")
        return None

def load_mtl(folder_path, epoch):
    """Loads a Multi-Task model using the Hybrid Engine"""
    checkpoint_path = os.path.join(folder_path, f"model_epoch_{epoch}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ MTL Checkpoint not found: {checkpoint_path}")
        return None

    backbone = get_backbone()
    
    configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    
    # Init Hybrid Engine
    engine = HybridEngine(backbone, configs, [], [], batch_size=BATCH_SIZE, device=DEVICE)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        engine.load_state_dict(state_dict)
        engine.eval()
        engine.to(DEVICE)
        print(f"✅ Loaded MTL Model from {folder_path} (Epoch {epoch})")
        return engine
    except Exception as e:
        print(f"❌ Error loading MTL: {e}")
        return None

# --- ANALYSIS 1: CORRELATION PLOT ---
def plot_correlations():
    print("\n⚗️ [Analysis] Generating Correlation Plots...")
    
    # 1. Get Data
    ds_thermo = Thermostability(verbose=0)
    _, _, thermo_test = ds_thermo.split()
    
    loader = torch.utils.data.DataLoader(thermo_test, batch_size=32, collate_fn=lambda b: [x['sequence'] for x in b])
    
    # Limit samples for speed
    seqs = []
    for b in loader:
        seqs.extend(b)
        if len(seqs) > 250: break
    
    data_store = {} # Store (x, y) tuples

    # --- A. Get Baseline Predictions ---
    model_t = load_baseline('Thermostability', PATH_BASE_THERMO)
    model_c = load_baseline('Cloning', PATH_BASE_CLONING)
    
    if model_t and model_c:
        x, y = [], []
        with torch.no_grad():
            for i in range(0, len(seqs), BATCH_SIZE):
                batch = seqs[i:i+BATCH_SIZE]
                # Tokenize for each model separately
                fmt = [" ".join(list(s)) for s in batch]
                
                # Thermo
                inp_t = model_t.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                emb_t = model_t.backbone(inp_t['input_ids'], inp_t['attention_mask'], task_type='sequence')
                pred_t = model_t.heads[0](emb_t).flatten().cpu().numpy() # Head 0 is the only head
                x.extend(pred_t)
                
                # Cloning
                inp_c = model_c.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                emb_c = model_c.backbone(inp_c['input_ids'], inp_c['attention_mask'], task_type='sequence')
                pred_c = torch.softmax(model_c.heads[0](emb_c), dim=1)[:, 1].cpu().numpy() # Head 0 is the only head
                y.extend(pred_c)
        data_store['Independent Baselines'] = (x, y)

    # --- B. Get MTL Predictions ---
    # Load Epoch 5 for UW (to show overfitting/noise) and Epoch 4/5 for PCGrad
    # Adjust epoch numbers based on what files you actually have!
    models = [
        ("Uncertainty Weighting", PATH_UW_DIR, 5), 
        ("Hybrid PCGrad", PATH_PC_DIR, 4) # Assuming PCGrad ran 4 epochs so far
    ]
    
    for name, path, ep in models:
        model = load_mtl(path, ep)
        if model:
            x, y = [], []
            with torch.no_grad():
                for i in range(0, len(seqs), BATCH_SIZE):
                    batch = seqs[i:i+BATCH_SIZE]
                    fmt = [" ".join(list(s)) for s in batch]
                    inputs = model.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                    emb = model.backbone(inputs['input_ids'], inputs['attention_mask'], task_type='sequence')
                    
                    # Head 0 = Thermo, Head 2 = Cloning
                    x.extend(model.heads[0](emb).flatten().cpu().numpy())
                    y.extend(torch.softmax(model.heads[2](emb), dim=1)[:, 1].cpu().numpy())
            data_store[name] = (x, y)

    # --- PLOT ---
    if not data_store: return

    fig, axes = plt.subplots(1, len(data_store), figsize=(6 * len(data_store), 5), sharey=True)
    if len(data_store) == 1: axes = [axes]
    
    for ax, (name, (x, y)) in zip(axes, data_store.items()):
        # Calculate correlation
        if len(x) > 1:
            corr, _ = pearsonr(x, y)
        else: corr = 0
        
        sns.scatterplot(x=x, y=y, alpha=0.6, ax=ax, hue=y, palette="viridis", legend=False)
        
        # Regression line
        try: sns.regplot(x=np.array(x), y=np.array(y), scatter=False, ax=ax, color='black', line_kws={'alpha':0.3})
        except: pass

        ax.set_title(f"{name}\nCorrelation r = {corr:.3f}", fontsize=14)
        ax.set_xlabel("Predicted Stability (Z-Score)")
        if name == list(data_store.keys())[0]: ax.set_ylabel("Pred. Solubility Probability")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "comparison_plots/stability_solubility_correlation_comparison.png"), dpi=300)
    plt.show()

# --- ANALYSIS 2: TRAJECTORY PLOT (Fixed for Lengths) ---
def plot_loss_trajectory():
    print("\n📊 [Analysis] Generating Trajectory Plots...")
    
    def get_df(folder):
        p = os.path.join(folder, "training_history.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    df_uw = get_df(PATH_UW_DIR)
    df_pc = get_df(PATH_PC_DIR)
    
    if df_uw is None or df_pc is None:
        print("❌ Missing history CSVs.")
        return

    # --- FIX: Truncate to common length ---
    min_len = min(len(df_uw), len(df_pc))
    df_uw = df_uw.iloc[:min_len]
    df_pc = df_pc.iloc[:min_len]
    
    epochs = range(1, min_len + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cloning (The Overfitting Test)
    ax1.plot(epochs, df_uw['Val_Loss_Cloning'], 'r--o', label='Uncertainty Weighting', linewidth=1.5)
    ax1.plot(epochs, df_pc['Val_Loss_Cloning'], 'g-o', label='Hybrid PCGrad', linewidth=2.5)
    ax1.set_title("Cloning Task: Overfitting Control")
    ax1.set_ylabel("Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Thermo (The Stability Test)
    ax2.plot(epochs, df_uw['Val_Loss_Thermostability'], 'r--o', label='Uncertainty Weighting', linewidth=1.5)
    ax2.plot(epochs, df_pc['Val_Loss_Thermostability'], 'g-o', label='Hybrid PCGrad', linewidth=2.5)
    ax2.set_title("Thermostability Task: MSE")
    ax2.set_ylabel("Validation MSE")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "comparison_plots/loss_trajectory_comparison.png"), dpi=300)
    plt.show()

# --- RUN ---
if __name__ == "__main__":
    # 1. Run Correlations (Uses weights)
    plot_correlations()
    
    # 2. Run Trajectories (Uses CSVs)
    plot_loss_trajectory()