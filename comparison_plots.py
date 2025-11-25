import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_with_task_specific_encoder import MultiTaskEngine  # For Baselines
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid       # For UW / PCGrad

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
LORA_RANK = 16
UNFROZEN_LAYERS = 0

BASE_DIR = "/content/drive/MyDrive/protein_multitask_outputs"

EXPERIMENTS = {
    "Baseline (Separate)": {
        "Thermo": os.path.join(BASE_DIR, "baseline_Thermostability_lora_only/model_epoch_5.pt"),
        "Cloning": os.path.join(BASE_DIR, "baseline_Cloning_lora_only/model_epoch_5.pt"),
        "Engine": "Standard"
    },
    "Uncertainty Weighting": {
        "Path": os.path.join(BASE_DIR, "cyclic_v1_lora16_uncertainty"),
        "Checkpoint": "model_epoch_5.pt", 
        "Engine": "Hybrid"
    },
    "Hybrid PCGrad": {
        "Path": os.path.join(BASE_DIR, "cyclic_v1_lora16_hybrid_pcgrad"),
        "Checkpoint": "model_epoch_4.pt", 
        "Engine": "Hybrid"
    }
}

def load_model_for_eval(config_key, specific_path=None):
    exp_config = EXPERIMENTS[config_key]
    engine_type = exp_config["Engine"]
    
    path = specific_path if specific_path else os.path.join(exp_config["Path"], exp_config["Checkpoint"])
    
    if not os.path.exists(path):
        print(f"⚠️ Warning: {path} not found. Skipping.")
        return None

    backbone = SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=UNFROZEN_LAYERS)
    
    if config_key == "Baseline (Separate)":
        # Baselines were trained with list-of-1 config. 
        # But to load weights, we need to match the ARCHITECTURE.
        # If you trained Thermo separately, it has heads.0 (Thermo).
        # If you trained Cloning separately, it has heads.0 (Cloning).
        # This is tricky. We need to know WHICH baseline we are loading.
        if "Thermostability" in path:
            configs = [{'name': 'Thermostability', 'type': 'regression', 'num_labels': 1}]
        elif "Cloning" in path:
            configs = [{'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}]
        else:
            return None # Unknown
    else:
        # MTL models have all 3
        configs = [
            {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
            {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
            {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
        ]

    # Initialize correct engine
    if engine_type == "Standard":
        # Pass dummy datasets
        engine = MultiTaskEngine(backbone, configs, [], [], batch_size=BATCH_SIZE, device=DEVICE)
    else:
        engine = MultiTaskEngineHybrid(backbone, configs, [], [], batch_size=BATCH_SIZE, device=DEVICE)
    
    try:
        state_dict = torch.load(path, map_location=DEVICE)
        # strict=False is vital because baselines are missing heads for other tasks
        engine.load_state_dict(state_dict, strict=False) 
        engine.eval()
        engine.to(DEVICE)
        return engine
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

# --- DATA LOADER ---
print("⏳ Loading Test Data...")
ds_thermo = Thermostability(verbose=0); _, _, thermo_test = ds_thermo.split()

# =========================================================
# ANALYSIS 1: CORRELATION (Physics)
# =========================================================
def plot_comparative_correlation():
    print("\n⚗️ [Analysis 1] Comparative Stability-Solubility Correlation")
    
    loader = torch.utils.data.DataLoader(thermo_test, batch_size=32, collate_fn=lambda b: [x['sequence'] for x in b])
    seqs_all = []
    for b in loader: 
        seqs_all.extend(b)
        if len(seqs_all) > 200: break 
    
    results = {}
    
    # 1. BASELINES (Need to load 2 models)
    model_t = load_model_for_eval("Baseline (Separate)", EXPERIMENTS["Baseline (Separate)"]["Thermo"])
    model_c = load_model_for_eval("Baseline (Separate)", EXPERIMENTS["Baseline (Separate)"]["Cloning"])
    
    if model_t and model_c:
        t_preds, c_probs = [], []
        with torch.no_grad():
            for i in range(0, len(seqs_all), BATCH_SIZE):
                batch = seqs_all[i:i+BATCH_SIZE]
                fmt = [" ".join(list(s)) for s in batch]
                
                # Thermo Model (Head 0)
                inp_t = model_t.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                emb_t = model_t.backbone(inp_t['input_ids'], inp_t['attention_mask'], task_type='sequence')
                t_preds.extend(model_t.heads[0](emb_t).flatten().cpu().numpy())
                
                # Cloning Model (Head 0 - Note: In baseline, it's the only head, so it's index 0!)
                inp_c = model_c.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                emb_c = model_c.backbone(inp_c['input_ids'], inp_c['attention_mask'], task_type='sequence')
                c_probs.extend(torch.softmax(model_c.heads[0](emb_c), dim=1)[:, 1].cpu().numpy())
        results['Baseline (Separate)'] = (t_preds, c_probs)

    # 2. MTL MODELS
    for key in ["Uncertainty Weighting", "Hybrid PCGrad"]:
        model = load_model_for_eval(key)
        if model:
            t_preds, c_probs = [], []
            with torch.no_grad():
                for i in range(0, len(seqs_all), BATCH_SIZE):
                    batch = seqs_all[i:i+BATCH_SIZE]
                    fmt = [" ".join(list(s)) for s in batch]
                    inputs = model.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                    emb = model.backbone(inputs['input_ids'], inputs['attention_mask'], task_type='sequence')
                    
                    # Head 0 (Thermo), Head 2 (Cloning)
                    t_preds.extend(model.heads[0](emb).flatten().cpu().numpy())
                    c_probs.extend(torch.softmax(model.heads[2](emb), dim=1)[:, 1].cpu().numpy())
            results[key] = (t_preds, c_probs)

    # Plot
    if not results: return
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5), sharey=True)
    if len(results) == 1: axes = [axes]
    
    for ax, (name, (x, y)) in zip(axes, results.items()):
        corr, _ = pearsonr(x, y)
        sns.scatterplot(x=x, y=y, alpha=0.6, ax=ax, hue=y, palette="viridis", legend=False)
        ax.set_title(f"{name}\nCorr: {corr:.3f}", fontsize=14)
        ax.set_xlabel("Predicted Stability")
        if name == list(results.keys())[0]: ax.set_ylabel("Pred. Solubility Prob")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "comparison_plots/comparative_stability_solubility_correlation.png"), dpi=300)
    plt.show()

# =========================================================
# ANALYSIS 2: TRAJECTORY COMPARISON (Loss Curves)
# =========================================================
def plot_trajectories():
    print("\n📊 [Analysis 2] Training Trajectory Comparison")
    
    def read_log(folder):
        p = os.path.join(folder, "training_history.csv")
        if os.path.exists(p): return pd.read_csv(p)
        return None

    df_uw = read_log(EXPERIMENTS["Uncertainty Weighting"]["Path"])
    df_pc = read_log(EXPERIMENTS["Hybrid PCGrad"]["Path"])
    
    if df_uw is None or df_pc is None:
        print("❌ CSV logs missing.")
        return

    # FIX: Truncate to common length
    min_len = min(len(df_uw), len(df_pc))
    df_uw = df_uw.iloc[:min_len]
    df_pc = df_pc.iloc[:min_len]
    epochs = df_uw['Epoch']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cloning
    ax1.plot(epochs, df_uw['Val_Loss_Cloning'], 'r--o', label='Uncertainty Weighting')
    ax1.plot(epochs, df_pc['Val_Loss_Cloning'], 'g-o', label='Hybrid PCGrad', linewidth=2.5)
    ax1.set_title("Cloning: Overfitting Control")
    ax1.set_ylabel("Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Thermo
    ax2.plot(epochs, df_uw['Val_Loss_Thermostability'], 'r--o', label='Uncertainty Weighting')
    ax2.plot(epochs, df_pc['Val_Loss_Thermostability'], 'g-o', label='Hybrid PCGrad', linewidth=2.5)
    ax2.set_title("Thermostability MSE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, "comparison_plots/training_trajectories_comparison.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_comparative_correlation()
    plot_trajectories()