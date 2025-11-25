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
PLOT_DIR = os.path.join(BASE_DIR, "final_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

EXPERIMENTS = {
    "Baseline": {
        "Thermo": os.path.join(BASE_DIR, "baseline_Thermostability_lora_only/model_epoch_5.pt"),
        "Cloning": os.path.join(BASE_DIR, "baseline_Cloning_lora_only/model_epoch_5.pt"),
    },
    "Uncertainty Weighting": os.path.join(BASE_DIR, "cyclic_v1_lora16_uncertainty/model_epoch_5.pt"),
    "Hybrid PCGrad": os.path.join(BASE_DIR, "cyclic_v1_lora16_hybrid_pcgrad/model_epoch_4.pt") # Or 5
}

# --- HELPERS ---
def get_backbone():
    return SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=UNFROZEN_LAYERS)

def load_baseline(task_name, path):
    if not os.path.exists(path): return None
    backbone = get_backbone()
    # Config must match training exactly (List of 1)
    if task_name == 'Thermo': cfg = [{'name': 'Thermostability', 'type': 'regression', 'num_labels': 1}]
    else: cfg = [{'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}]
    
    engine = StandardEngine(backbone, cfg, [], [], batch_size=BATCH_SIZE, device=DEVICE)
    state_dict = torch.load(path, map_location=DEVICE)
    engine.load_state_dict(state_dict) # Strict=True is fine for StandardEngine loading Standard weights
    engine.eval(); engine.to(DEVICE)
    return engine

def load_mtl(path):
    if not os.path.exists(path): return None
    backbone = get_backbone()
    cfg = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    engine = HybridEngine(backbone, cfg, [], [], batch_size=BATCH_SIZE, device=DEVICE)
    state_dict = torch.load(path, map_location=DEVICE)
    engine.load_state_dict(state_dict, strict=False) 
    engine.eval(); engine.to(DEVICE)
    return engine

# --- DATA GENERATION ---
def generate_predictions():
    print("⏳ Loading Test Data & Models...")
    ds = Thermostability(verbose=0); _, _, test_set = ds.split()
    loader = torch.utils.data.DataLoader(test_set, batch_size=32, collate_fn=lambda b: [x['sequence'] for x in b])
    
    # Collect Sequences
    seqs = []
    for b in loader:
        seqs.extend(b)
        if len(seqs) > 300: break # Analyze 300 proteins
    
    # Load Models
    m_base_t = load_baseline('Thermo', EXPERIMENTS["Baseline"]["Thermo"])
    m_base_c = load_baseline('Cloning', EXPERIMENTS["Baseline"]["Cloning"])
    m_uw = load_mtl(EXPERIMENTS["Uncertainty Weighting"])
    m_pc = load_mtl(EXPERIMENTS["Hybrid PCGrad"])
    
    data = {
        'Baseline': {'Thermo': [], 'Cloning': []},
        'UW': {'Thermo': [], 'Cloning': []},
        'PCGrad': {'Thermo': [], 'Cloning': []}
    }
    
    print("⚡ Running Inference...")
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            batch = seqs[i:i+BATCH_SIZE]
            fmt = [" ".join(list(s)) for s in batch]
            
            # 1. Baseline
            if m_base_t and m_base_c:
                inp_t = m_base_t.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                emb_t = m_base_t.backbone(inp_t['input_ids'], inp_t['attention_mask'], task_type='sequence')
                data['Baseline']['Thermo'].extend(m_base_t.heads[0](emb_t).flatten().cpu().numpy())
                
                inp_c = m_base_c.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                emb_c = m_base_c.backbone(inp_c['input_ids'], inp_c['attention_mask'], task_type='sequence')
                data['Baseline']['Cloning'].extend(torch.softmax(m_base_c.heads[0](emb_c), dim=1)[:, 1].cpu().numpy())

            # 2. MTL Models
            for name, model in [('UW', m_uw), ('PCGrad', m_pc)]:
                if model:
                    inp = model.backbone.tokenizer(fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                    emb = model.backbone(inp['input_ids'], inp['attention_mask'], task_type='sequence')
                    data[name]['Thermo'].extend(model.heads[0](emb).flatten().cpu().numpy())
                    data[name]['Cloning'].extend(torch.softmax(model.heads[2](emb), dim=1)[:, 1].cpu().numpy())

    return data

# =========================================================
# PLOT 1: MODEL vs MODEL SCATTER (The "Correction" Plot)
# =========================================================
def plot_prediction_shift(data):
    print("\n📊 [Plot 1] Generating Model vs Model Shift...")
    
    if not data['Baseline']['Thermo'] or not data['PCGrad']['Thermo']:
        print("❌ Missing data for comparison.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Thermostability Shift
    x = data['Baseline']['Thermo']
    y = data['PCGrad']['Thermo']
    
    sns.scatterplot(x=x, y=y, ax=axes[0], alpha=0.6, color='teal')
    # Draw diagonal identity line
    lims = [min(min(x), min(y)), max(max(x), max(y))]
    axes[0].plot(lims, lims, 'r--', alpha=0.75, label="No Change")
    
    axes[0].set_title("Thermostability: Baseline vs PCGrad", fontsize=14)
    axes[0].set_xlabel("Baseline Prediction (Single Task)")
    axes[0].set_ylabel("PCGrad Prediction (Multi Task)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Subplot 2: Cloning Shift
    x = data['Baseline']['Cloning']
    y = data['PCGrad']['Cloning']
    
    sns.scatterplot(x=x, y=y, ax=axes[1], alpha=0.6, color='purple')
    axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.75, label="No Change")
    
    axes[1].set_title("Solubility Probability: Baseline vs PCGrad", fontsize=14)
    axes[1].set_xlabel("Baseline Prediction")
    axes[1].set_ylabel("PCGrad Prediction")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "model_prediction_shift.png"), dpi=300)
    plt.show()
    print("   -> Interpretation: Points OFF the red line indicate proteins where PCGrad 'disagrees' with the Baseline due to multi-task constraints.")

# =========================================================
# PLOT 2: PREDICTION DISTRIBUTIONS (The "Regularization" Plot)
# =========================================================
def plot_distributions(data):
    print("\n📊 [Plot 2] Generating Distribution KDEs...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Thermo Dist
    sns.kdeplot(data['Baseline']['Thermo'], ax=axes[0], label='Baseline', fill=True, alpha=0.2)
    if data['UW']['Thermo']: sns.kdeplot(data['UW']['Thermo'], ax=axes[0], label='UW', fill=True, alpha=0.2)
    sns.kdeplot(data['PCGrad']['Thermo'], ax=axes[0], label='PCGrad', fill=True, alpha=0.2)
    axes[0].set_title("Thermostability Prediction Density")
    axes[0].set_xlabel("Predicted Z-Score")
    axes[0].legend()
    
    # Cloning Dist
    sns.kdeplot(data['Baseline']['Cloning'], ax=axes[1], label='Baseline', fill=True, alpha=0.2)
    if data['UW']['Cloning']: sns.kdeplot(data['UW']['Cloning'], ax=axes[1], label='UW', fill=True, alpha=0.2)
    sns.kdeplot(data['PCGrad']['Cloning'], ax=axes[1], label='PCGrad', fill=True, alpha=0.2)
    axes[1].set_title("Solubility Probability Density")
    axes[1].set_xlabel("Probability")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "prediction_distributions.png"), dpi=300)
    plt.show()
    print("   -> Interpretation: Tighter distributions suggest regularization. Bimodal distributions suggest strong classification confidence.")

if __name__ == "__main__":
    data = generate_predictions()
    if data:
        plot_prediction_shift(data)
        plot_distributions(data)