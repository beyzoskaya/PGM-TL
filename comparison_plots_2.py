import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from engine_hf_with_task_specific_encoder import MultiTaskEngine as StandardEngine
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid as HybridEngine
from flip_hf import SecondaryStructure, Thermostability, CloningCLF
from protbert_hf import SharedProtBert

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = "/content/drive/MyDrive/protein_multitask_outputs"
SAVE_DIR = os.path.join(BASE_DIR, "final_plots")
os.makedirs(SAVE_DIR, exist_ok=True)

PATHS = {
    "Baseline": {
        "SSP": os.path.join(BASE_DIR, "baseline_SecStructure_lora_only/model_epoch_5.pt"),
        "Thermo": os.path.join(BASE_DIR, "baseline_Thermostability_lora_only/model_epoch_5.pt"),
        "Cloning": os.path.join(BASE_DIR, "baseline_Cloning_lora_only/model_epoch_5.pt")
    },
    "PCGrad": os.path.join(BASE_DIR, "cyclic_v1_lora16_hybrid_pcgrad/model_epoch_4.pt") # or 5
}

KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def calculate_hydrophobicity(seq):
    """Calculates GRAVY score (Grand Average of Hydropathy)."""
    scores = [KYTE_DOOLITTLE.get(aa, 0) for aa in seq]
    return sum(scores) / (len(scores) + 1e-9)

# Q8 to Q3 Mapping (Standard Reduction)
# 0:G(H), 1:H(H), 2:I(H), 3:T(C), 4:E(E), 5:B(E), 6:S(C), 7:C(C)
# H=Helix, E=Sheet, C=Coil
Q8_TO_Q3 = {0: 'Helix', 1: 'Helix', 2: 'Helix', 
            3: 'Coil', 6: 'Coil', 7: 'Coil', 
            4: 'Sheet', 5: 'Sheet'}

def load_pcgrad():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    cfg = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    engine = HybridEngine(backbone, cfg, [], [], batch_size=1, device=DEVICE)
    engine.load_state_dict(torch.load(PATHS["PCGrad"], map_location=DEVICE), strict=False)
    engine.eval().to(DEVICE)
    return engine

def load_baseline_ssp():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    cfg = [{'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8}]
    engine = StandardEngine(backbone, cfg, [], [], batch_size=1, device=DEVICE)
    engine.load_state_dict(torch.load(PATHS["Baseline"]["SSP"], map_location=DEVICE))
    engine.eval().to(DEVICE)
    return engine

def load_baseline_cloning():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    cfg = [{'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}]
    engine = StandardEngine(backbone, cfg, [], [], batch_size=1, device=DEVICE)
    engine.load_state_dict(torch.load(PATHS["Baseline"]["Cloning"], map_location=DEVICE))
    engine.eval().to(DEVICE)
    return engine

# --- DATA GATHERING ---
def run_analysis():
    print("--- Loading Data & Models ---")
    
    # 1. Secondary Structure Analysis Data
    ds_ssp = SecondaryStructure(verbose=0)
    _, _, test_ssp = ds_ssp.split()
    
    true_labels_q3 = []
    pred_base_q3 = []
    pred_pc_q3 = []

    model_pc = load_pcgrad()
    model_base_ssp = load_baseline_ssp()

    print("Running SSP Inference...")
    for i in tqdm(range(min(200, len(test_ssp)))): # First 200 proteins
        item = test_ssp[i]
        seq = " ".join(list(item['sequence']))
        targets = item['targets']['target'] # List of ints
        
        # PCGrad
        inp = model_pc.backbone.tokenizer([seq], return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
        out_pc = model_pc.heads[1](model_pc.backbone(inp['input_ids'], inp['attention_mask'], 'token'))
        preds_pc = out_pc.argmax(dim=-1).squeeze().cpu().numpy()
        
        # Baseline
        out_base = model_base_ssp.heads[0](model_base_ssp.backbone(inp['input_ids'], inp['attention_mask'], 'token'))
        preds_base = out_base.argmax(dim=-1).squeeze().cpu().numpy()
        
        # Align lengths (Tokens vs Raw Seq)
        # We slice [1:-1] to remove CLS/SEP, then truncate to valid len
        valid_len = min(len(targets), len(preds_pc)-2)
        
        p_pc_clean = preds_pc[1 : 1+valid_len]
        p_base_clean = preds_base[1 : 1+valid_len]
        t_clean = targets[:valid_len]
        
        for p_p, p_b, t in zip(p_pc_clean, p_base_clean, t_clean):
            if t != -100: # Ignore padding
                true_labels_q3.append(Q8_TO_Q3.get(t, 'Coil'))
                pred_pc_q3.append(Q8_TO_Q3.get(p_p, 'Coil'))
                pred_base_q3.append(Q8_TO_Q3.get(p_b, 'Coil'))

    # 2. Solubility vs Hydrophobicity Data
    ds_cloning = CloningCLF(verbose=0)
    _, _, test_cloning = ds_cloning.split()
    model_base_clone = load_baseline_cloning()
    
    solubility_data = []
    print("Running Solubility Inference...")
    for i in tqdm(range(min(200, len(test_cloning)))):
        item = test_cloning[i]
        seq_str = item['sequence']
        seq_fmt = " ".join(list(seq_str))
        
        gravy = calculate_hydrophobicity(seq_str)
        
        inp = model_pc.backbone.tokenizer([seq_fmt], return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
        
        # PCGrad Prob
        emb_pc = model_pc.backbone(inp['input_ids'], inp['attention_mask'], 'sequence')
        prob_pc = torch.softmax(model_pc.heads[2](emb_pc), dim=1)[0, 1].item()
        
        # Baseline Prob
        emb_base = model_base_clone.backbone(inp['input_ids'], inp['attention_mask'], 'sequence')
        prob_base = torch.softmax(model_base_clone.heads[0](emb_base), dim=1)[0, 1].item()
        
        solubility_data.append({"GRAVY": gravy, "Prob": prob_pc, "Model": "PCGrad"})
        solubility_data.append({"GRAVY": gravy, "Prob": prob_base, "Model": "Baseline"})

    return {
        "ssp": (true_labels_q3, pred_base_q3, pred_pc_q3),
        "sol": pd.DataFrame(solubility_data)
    }

# --- PLOTTING ---
def plot_results(data):
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    
    # === PLOT 1: Confusion Matrix (Structural Fidelity) ===
    labels = ['Helix', 'Sheet', 'Coil']
    true_l, pred_base, pred_pc = data['ssp']
    
    cm_base = confusion_matrix(true_l, pred_base, labels=labels, normalize='true')
    cm_pc = confusion_matrix(true_l, pred_pc, labels=labels, normalize='true')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(cm_base, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Baseline: Structural Confusion")
    axes[0].set_ylabel("True Structure")
    axes[0].set_xlabel("Predicted Structure")
    
    sns.heatmap(cm_pc, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap="Greens", ax=axes[1], cbar=False)
    axes[1].set_title("PCGrad: Structural Confusion")
    axes[1].set_xlabel("Predicted Structure")
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "1_Bio_Structure_Confusion.png"), dpi=300)
    print("Saved Plot 1")

    # === PLOT 2: Solubility vs Hydrophobicity (Chemical Logic) ===
    df_sol = data['sol']
    plt.figure(figsize=(10, 7))
    
    # We use a regression plot to show the trend lines
    sns.lmplot(data=df_sol, x="GRAVY", y="Prob", hue="Model", palette={"Baseline": "gray", "PCGrad": "#1D3557"},
               scatter_kws={'alpha': 0.4}, height=6, aspect=1.3)
    
    plt.title("Chemical Logic Check: Hydrophobicity vs. Solubility")
    plt.xlabel("GRAVY Score (More Positive = More Hydrophobic)")
    plt.ylabel("Predicted Solubility Probability")
    plt.axvline(0, color='k', linestyle='--', alpha=0.2)
    plt.text(0.5, 0.9, "Hydrophobic\n(Should be Insoluble)", color='red', transform=plt.gca().transAxes)
    plt.text(0.05, 0.9, "Hydrophilic\n(Should be Soluble)", color='green', transform=plt.gca().transAxes)
    
    plt.savefig(os.path.join(SAVE_DIR, "2_Bio_Chemical_Logic.png"), dpi=300)
    print("Saved Plot 2")

if __name__ == "__main__":
    data = run_analysis()
    plot_results(data)