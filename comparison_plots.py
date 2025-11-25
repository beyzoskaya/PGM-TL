import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
LORA_RANK = 16
UNFROZEN_LAYERS = 0

BASE_DIR = "/content/drive/MyDrive/protein_multitask_outputs"
PATH_UW = os.path.join(BASE_DIR, "cyclic_v1_lora16_uncertainty")
PATH_PCGRAD = os.path.join(BASE_DIR, "cyclic_v1_lora16_hybrid_pcgrad")

PATH_BASE_THERMO = os.path.join(BASE_DIR, "baseline_Thermostability_lora_only/model_epoch_5.pt") 
PATH_BASE_CLONING = os.path.join(BASE_DIR, "baseline_Cloning_lora_only/model_epoch_5.pt")       

def load_engine(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Warning: {checkpoint_path} not found. Skipping.")
        return None

    backbone = SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=UNFROZEN_LAYERS)
    
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    
    engine = MultiTaskEngineHybrid(backbone, task_configs, [], [], batch_size=BATCH_SIZE, device=DEVICE)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        engine.load_state_dict(state_dict, strict=False) # strict=False handles missing log_vars in baselines
        engine.eval()
        engine.to(DEVICE)
        return engine
    except Exception as e:
        print(f"❌ Error loading {checkpoint_path}: {e}")
        return None

print("⏳ Loading Test Data...")
ds_thermo = Thermostability(verbose=0); _, _, thermo_test = ds_thermo.split()
# We use Thermo test set for correlations because it has "Stability" ground truth labels if needed
# but mainly we need sequences to predict both properties.

# =========================================================
# ANALYSIS 1: PHYSICAL COUPLING (CORRELATION SCATTER)
# =========================================================
def plot_comparative_correlation():
    print("\n⚗️ [Analysis 1] Comparative Stability-Solubility Correlation")
    
    # 1. Get Sequences
    loader = torch.utils.data.DataLoader(thermo_test, batch_size=32, collate_fn=lambda b: [x['sequence'] for x in b])
    seqs_all = []
    for b in loader: 
        seqs_all.extend(b)
        if len(seqs_all) > 200: break # Limit to 200 proteins
    
    results = {}
    
    # --- A. Single Task Baselines (Two separate models) ---
    model_t = load_engine(PATH_BASE_THERMO)
    model_c = load_engine(PATH_BASE_CLONING)
    
    if model_t and model_c:
        t_preds, c_probs = [], []
        with torch.no_grad():
            for i in range(0, len(seqs_all), BATCH_SIZE):
                batch = seqs_all[i:i+BATCH_SIZE]
                fmt = [" ".join(list(s)) for s in batch]
                
                # Thermo Model
                inp_t = model_t.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                emb_t = model_t.backbone(inp_t['input_ids'], inp_t['attention_mask'], task_type='sequence')
                t_preds.extend(model_t.heads[0](emb_t).flatten().cpu().numpy())
                
                # Cloning Model
                inp_c = model_c.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                emb_c = model_c.backbone(inp_c['input_ids'], inp_c['attention_mask'], task_type='sequence')
                c_probs.extend(torch.softmax(model_c.heads[2](emb_c), dim=1)[:, 1].cpu().numpy())
        
        results['Baseline (Separate)'] = (t_preds, c_probs)

    # --- B. Multi-Task Models (Epoch 5 for UW vs PCGrad) ---
    for name, path in [("UW (Epoch 5)", os.path.join(PATH_UW, "model_epoch_5.pt")), 
                       ("PCGrad (Epoch 5)", os.path.join(PATH_PCGRAD, "model_epoch_4.pt"))]:
        model = load_engine(path)
        if model:
            t_preds, c_probs = [], []
            with torch.no_grad():
                for i in range(0, len(seqs_all), BATCH_SIZE):
                    batch = seqs_all[i:i+BATCH_SIZE]
                    fmt = [" ".join(list(s)) for s in batch]
                    inputs = model.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                    emb = model.backbone(inputs['input_ids'], inputs['attention_mask'], task_type='sequence')
                    
                    t_preds.extend(model.heads[0](emb).flatten().cpu().numpy())
                    c_probs.extend(torch.softmax(model.heads[2](emb), dim=1)[:, 1].cpu().numpy())
            results[name] = (t_preds, c_probs)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for ax, (name, (x, y)) in zip(axes, results.items()):
        corr, _ = pearsonr(x, y)
        sns.scatterplot(x=x, y=y, alpha=0.6, ax=ax, hue=y, palette="viridis", legend=False)
        sns.regplot(x=x, y=y, scatter=False, ax=ax, color='black', line_kws={'alpha':0.5})
        ax.set_title(f"{name}\nCorr: {corr:.3f}", fontsize=14)
        ax.set_xlabel("Predicted Stability")
        if name == "Baseline (Separate)": ax.set_ylabel("Pred. Solubility Prob")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Evolution of Physical Understanding: Independent vs Joint Learning", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "comparison_plots/comparative_stability_solubility_correlation.png"), dpi=300)
    plt.show()

# =========================================================
# ANALYSIS 2: DMS HEATMAP COMPARISON
# =========================================================
def plot_comparative_dms():
    print("\n🔬 [Analysis 2] Comparative Deep Mutational Scanning")
    
    # Find long protein
    target_seq = next((item['sequence'] for item in thermo_test if 200 < len(item['sequence']) < 300), thermo_test[0]['sequence'])
    print(f"   Protein Length: {len(target_seq)}")
    
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    
    # Generate Mutants
    mutants_seqs = []
    coords = []
    for p_i in range(len(target_seq)):
        for a_i, aa in enumerate(amino_acids):
            m = list(target_seq); m[p_i] = aa
            mutants_seqs.append(" ".join(m))
            coords.append((a_i, p_i))
            
    def get_heatmap(model_path):
        model = load_engine(model_path)
        if not model: return None
        preds = []
        with torch.no_grad():
            for i in range(0, len(mutants_seqs), 64):
                batch = mutants_seqs[i:i+64]
                inp = model.backbone.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(DEVICE)
                emb = model.backbone(inp['input_ids'], inp['attention_mask'], task_type='sequence')
                preds.extend(model.heads[0](emb).flatten().cpu().numpy())
        
        hm = np.zeros((20, len(target_seq)))
        for p, (r, c) in zip(preds, coords): hm[r, c] = p
        return hm

    # Load 3 Maps
    hm_base = get_heatmap(PATH_BASE_THERMO)
    hm_uw = get_heatmap(os.path.join(PATH_UW, "model_epoch_5.pt"))
    hm_pc = get_heatmap(os.path.join(PATH_PCGRAD, "model_epoch_4.pt"))

    if hm_base is None or hm_pc is None: return

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    cbar_ax = fig.add_axes([.91, .3, .02, .4]) # Common colorbar
    
    # Determine common scale
    vmin = min(hm_base.min(), hm_pc.min())
    vmax = max(hm_base.max(), hm_pc.max())

    sns.heatmap(hm_base, ax=axes[0], cmap="coolwarm", vmin=vmin, vmax=vmax, cbar=False, yticklabels=amino_acids)
    axes[0].set_title("Baseline (Single Task Thermostability)")
    
    if hm_uw is not None:
        sns.heatmap(hm_uw, ax=axes[1], cmap="coolwarm", vmin=vmin, vmax=vmax, cbar=False, yticklabels=amino_acids)
        axes[1].set_title("Uncertainty Weighting (Epoch 5 - Likely Noisy/Overfit)")
    
    sns.heatmap(hm_pc, ax=axes[2], cmap="coolwarm", vmin=vmin, vmax=vmax, cbar_ax=cbar_ax, yticklabels=amino_acids)
    axes[2].set_title("Hybrid PCGrad (Epoch 5)")
    
    axes[2].set_xlabel("Residue Position")
    plt.savefig(os.path.join(BASE_DIR, "comparison_plots/comparative_dms_heatmaps.png"), dpi=300, bbox_inches='tight')
    plt.show()

# =========================================================
# ANALYSIS 3: TRAJECTORY COMPARISON
# =========================================================
def plot_trajectories():
    print("\n📊 [Analysis 3] Training Trajectory Comparison")
    
    # Helper to read CSV
    def read_log(folder):
        p = os.path.join(folder, "training_history.csv")
        if os.path.exists(p): return pd.read_csv(p)
        return None

    df_uw = read_log(PATH_UW)
    df_pc = read_log(PATH_PCGRAD)
    
    if df_uw is None or df_pc is None:
        print("❌ CSV logs missing.")
        return

    epochs = df_uw['Epoch']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cloning Val Loss (The Overfitting Indicator)
    ax1.plot(epochs, df_uw['Val_Loss_Cloning'], 'r--o', label='Uncertainty Weighting', alpha=0.7)
    ax1.plot(epochs, df_pc['Val_Loss_Cloning'], 'g-o', label='Hybrid PCGrad', linewidth=2.5)
    ax1.set_title("Cloning Task: Overfitting Control")
    ax1.set_ylabel("Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Thermostability Val Loss
    ax2.plot(epochs, df_uw['Val_Loss_Thermostability'], 'r--o', label='Uncertainty Weighting', alpha=0.7)
    ax2.plot(epochs, df_pc['Val_Loss_Thermostability'], 'g-o', label='Hybrid PCGrad', linewidth=2.5)
    
    # Add Baseline line if available (constant line)
    # Assuming baseline ended at 0.35 for example (update this value manually if you have it)
    # ax2.axhline(y=0.35, color='gray', linestyle=':', label='Single Task Baseline')
    
    ax2.set_title("Thermostability: Does MTL hurt?")
    ax2.set_ylabel("Validation MSE")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, "comparison_plots/training_trajectories_comparison.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_comparative_correlation()
    plot_comparative_dms()
    plot_trajectories()