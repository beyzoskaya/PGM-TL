import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from engine_hf_with_task_specific_encoder import MultiTaskEngine as StandardEngine
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid as HybridEngine
from flip_hf import Thermostability
from protbert_hf import SharedProtBert

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LORA_RANK = 16
UNFROZEN_LAYERS = 0 

BASE_DIR = "/content/drive/MyDrive/protein_multitask_outputs"
PLOT_DIR = os.path.join(BASE_DIR, "final_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

EXPERIMENTS = {
    "Baseline": {
        "Thermo": os.path.join(BASE_DIR, "baseline_Thermostability_lora_only/model_epoch_5.pt"),
        "Cloning": os.path.join(BASE_DIR, "baseline_Cloning_lora_only/model_epoch_5.pt"),
        "SecStructure": os.path.join(BASE_DIR, "baseline_SecStructure_lora_only/model_epoch_5.pt")
    },
    "Uncertainty Weighting": os.path.join(BASE_DIR, "cyclic_v1_lora16_uncertainty/model_epoch_5.pt"),
    "Hybrid PCGrad": os.path.join(BASE_DIR, "cyclic_v1_lora16_hybrid_pcgrad/model_epoch_4.pt") 
}

def get_backbone():
    return SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=UNFROZEN_LAYERS)

def load_single_task_model(task_type, path):
    """Loads a Baseline model (StandardEngine)"""
    if not os.path.exists(path):
        print(f"⚠️ Warning: File not found {path}")
        return None
    
    backbone = get_backbone()
    
    if task_type == 'Thermo':
        cfg = [{'name': 'Thermostability', 'type': 'regression', 'num_labels': 1}]
    elif task_type == 'Cloning':
        cfg = [{'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}]
    elif task_type == 'SecStructure':
        cfg = [{'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8}]
        
    # StandardEngine expects list of datasets, we pass empty lists just to initialize
    engine = StandardEngine(backbone, cfg, [], [], batch_size=1, device=DEVICE)
    
    # Load weights
    state_dict = torch.load(path, map_location=DEVICE)
    engine.load_state_dict(state_dict)
    engine.eval()
    engine.to(DEVICE)
    return engine

def load_multitask_model(path, mode="Hybrid"):
    """Loads UW or PCGrad model"""
    if not os.path.exists(path):
        print(f"⚠️ Warning: File not found {path}")
        return None
    
    backbone = get_backbone()
    
    # Order must match training script exactly
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    
    # Use HybridEngine class (works for both UW and PCGrad architectures in your setup)
    engine = HybridEngine(backbone, task_configs, [], [], batch_size=1, device=DEVICE)
    
    state_dict = torch.load(path, map_location=DEVICE)
    # strict=False allows loading even if some internal buffer keys differ slightly
    engine.load_state_dict(state_dict, strict=False) 
    engine.eval()
    engine.to(DEVICE)
    return engine

# --- INFERENCE ENGINE ---

def run_inference():
    print("--- 1. Loading Data ---")
    # We use Thermostability Test set as our "Probe" dataset 
    # (it contains sequences we want to analyze)
    ds = Thermostability(verbose=0)
    _, _, test_set = ds.split()
    
    # Limit to 200 proteins for speed, but enough for statistics
    subset_indices = range(min(200, len(test_set))) 
    sequences = [test_set[i]['sequence'] for i in subset_indices]
    print(f"Loaded {len(sequences)} sequences for biological analysis.")

    # Results Container
    results = []

    # --- 2. BASELINE INFERENCE ---
    print("\n--- 2. Running Baseline Models ---")
    m_base_t = load_single_task_model('Thermo', EXPERIMENTS["Baseline"]["Thermo"])
    m_base_c = load_single_task_model('Cloning', EXPERIMENTS["Baseline"]["Cloning"])
    m_base_s = load_single_task_model('SecStructure', EXPERIMENTS["Baseline"]["SecStructure"])

    with torch.no_grad():
        for i, seq in enumerate(tqdm(sequences, desc="Baseline")):
            fmt_seq = " ".join(list(seq)) # Space format for ProtBert
            
            # Thermo
            res_t = np.nan
            if m_base_t:
                inp = m_base_t.backbone.tokenizer([fmt_seq], return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
                emb = m_base_t.backbone(inp['input_ids'], inp['attention_mask'], task_type='sequence')
                res_t = m_base_t.heads[0](emb).item()
            
            # Cloning
            res_c = np.nan
            if m_base_c:
                inp = m_base_c.backbone.tokenizer([fmt_seq], return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
                emb = m_base_c.backbone(inp['input_ids'], inp['attention_mask'], task_type='sequence')
                logits = m_base_c.heads[0](emb)
                res_c = torch.softmax(logits, dim=1)[0, 1].item() # Prob of class 1 (Soluble)

            # SecStruct
            frac_helix = np.nan
            if m_base_s:
                inp = m_base_s.backbone.tokenizer([fmt_seq], return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
                emb = m_base_s.backbone(inp['input_ids'], inp['attention_mask'], task_type='token')
                logits = m_base_s.heads[0](emb) # [1, seq_len, 8]
                preds = logits.argmax(dim=-1).squeeze().cpu().numpy()
                # Assuming Label 0 or 1 is Helix (H) in Q8. 
                # We calculate % of residues predicted as class 0 (Proxy for Helix)
                frac_helix = np.sum(preds == 0) / len(preds)

            results.append({
                "Model": "Baseline",
                "Sequence_ID": i,
                "Stability": res_t,
                "Solubility_Prob": res_c,
                "Helix_Fraction": frac_helix
            })
    
    # Cleanup GPU
    del m_base_t, m_base_c, m_base_s
    torch.cuda.empty_cache()

    # --- 3. MULTITASK INFERENCE ---
    for model_name, path in [("Uncertainty", EXPERIMENTS["Uncertainty Weighting"]), ("PCGrad", EXPERIMENTS["Hybrid PCGrad"])]:
        print(f"\n--- 3. Running {model_name} Model ---")
        model = load_multitask_model(path)
        if not model: continue

        with torch.no_grad():
            for i, seq in enumerate(tqdm(sequences, desc=model_name)):
                fmt_seq = " ".join(list(seq))
                inp = model.backbone.tokenizer([fmt_seq], return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
                
                # 1. Stability (Head 0)
                emb_seq = model.backbone(inp['input_ids'], inp['attention_mask'], task_type='sequence')
                res_t = model.heads[0](emb_seq).item()
                
                # 2. SecStruct (Head 1) - Requires Token Embeddings
                emb_tok = model.backbone(inp['input_ids'], inp['attention_mask'], task_type='token')
                logits_s = model.heads[1](emb_tok)
                preds_s = logits_s.argmax(dim=-1).squeeze().cpu().numpy()
                frac_helix = np.sum(preds_s == 0) / len(preds_s)

                # 3. Cloning (Head 2)
                logits_c = model.heads[2](emb_seq)
                res_c = torch.softmax(logits_c, dim=1)[0, 1].item()

                results.append({
                    "Model": model_name,
                    "Sequence_ID": i,
                    "Stability": res_t,
                    "Solubility_Prob": res_c,
                    "Helix_Fraction": frac_helix
                })
        
        del model
        torch.cuda.empty_cache()

    return pd.DataFrame(results)

# --- PLOTTING FUNCTIONS ---

def plot_tradeoff_stability_solubility(df):
    """
    Biological Q: Do stable proteins tend to be soluble? 
    Does PCGrad find a better Pareto frontier?
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, 
        x="Stability", 
        y="Solubility_Prob", 
        hue="Model", 
        style="Model",
        alpha=0.7,
        palette="deep"
    )
    plt.title("Biological Landscape: Stability vs. Solubility Prediction")
    plt.xlabel("Predicted Thermostability (Z-Score)")
    plt.ylabel("Predicted Cloning Probability (Solubility)")
    
    # Add quadrants
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.text(2, 0.9, "Ideal Candidates\n(Stable & Soluble)", color='green', fontsize=9)
    plt.text(-2, 0.1, "Difficult Candidates\n(Unstable & Insoluble)", color='red', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "1_bio_tradeoff_stability_solubility.png"), dpi=300)
    print("Saved Plot 1: Stability vs Solubility")

def plot_structure_function_correlation(df):
    """
    Biological Q: Does the model learn that alpha-helices contribute to stability?
    """
    plt.figure(figsize=(10, 7))
    
    # Create bins for Helix Fraction to make the plot readable (Hexbin or Boxplot equivalent)
    # Using lmplot for trend lines
    sns.lmplot(
        data=df, 
        x="Helix_Fraction", 
        y="Stability", 
        hue="Model", 
        height=6, 
        aspect=1.5,
        scatter_kws={'alpha': 0.4},
        line_kws={'linewidth': 2}
    )
    
    plt.title("Structure-Property Relationship: Helix Content vs. Stability")
    plt.xlabel("Predicted Helix Fraction (Structural Rigidity)")
    plt.ylabel("Predicted Thermostability")
    
    plt.savefig(os.path.join(PLOT_DIR, "2_bio_structure_stability_corr.png"), dpi=300)
    print("Saved Plot 2: Structure vs Stability")

def plot_prediction_regularization(df):
    """
    Shows if Multi-Task Learning constrains the predictions compared to Baseline.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ["Stability", "Solubility_Prob", "Helix_Fraction"]
    titles = ["Thermostability Dist.", "Solubility Prob Dist.", "Helix Fraction Dist."]
    
    for i, metric in enumerate(metrics):
        sns.violinplot(data=df, x="Model", y=metric, ax=axes[i], palette="pastel")
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("")
    
    plt.suptitle("Impact of Multi-Task Learning on Prediction Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "3_prediction_distributions.png"), dpi=300)
    print("Saved Plot 3: Distributions")

def analyze_correlations(df):
    """
    Prints Pearson correlations between tasks for each model.
    """
    print("\n--- Biological Correlation Analysis ---")
    for model in df['Model'].unique():
        sub = df[df['Model'] == model]
        corr_stab_sol = sub['Stability'].corr(sub['Solubility_Prob'])
        corr_stab_helix = sub['Stability'].corr(sub['Helix_Fraction'])
        
        print(f"[{model}]")
        print(f"  Corr(Stability, Solubility): {corr_stab_sol:.3f}")
        print(f"  Corr(Stability, Helix %):    {corr_stab_helix:.3f}")
        print("  -> If PCGrad correlation > Baseline, MTL is learning shared physical constraints.")

# --- MAIN ---

if __name__ == "__main__":
    print(f"Starting Biological Analysis on {DEVICE}")
    
    # 1. Run Inference
    df_results = run_inference()
    
    # 2. Save Raw Data
    df_results.to_csv(os.path.join(PLOT_DIR, "inference_results.csv"), index=False)
    
    # 3. Generate Plots
    if not df_results.empty:
        plot_tradeoff_stability_solubility(df_results)
        plot_structure_function_correlation(df_results)
        plot_prediction_regularization(df_results)
        analyze_correlations(df_results)
        
        print(f"\nAll plots saved to: {PLOT_DIR}")
    else:
        print("No results generated. Check model paths.")