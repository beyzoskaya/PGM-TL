import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import r2_score, confusion_matrix, classification_report

from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = "/content/drive/MyDrive/protein_multitask_outputs"
SAVE_DIR = os.path.join(BASE_DIR, "final_plots/diagnostic_hybrid_pcgrad")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "cyclic_v1_lora16_hybrid_pcgrad/model_epoch_4.pt")

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    
    engine = MultiTaskEngineHybrid(backbone, task_configs, [], [], batch_size=1, device=DEVICE)
    # strict=False is safe here as long as architecture matches
    engine.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    engine.eval().to(DEVICE)
    return engine

def gather_test_data(engine):
    """Runs inference on ALL test sets and collects raw lists."""
    results = {
        "Thermo": {"true": [], "pred": []},
        "SSP":    {"true": [], "pred": []}, # Flattened
        "Clone":  {"true": [], "pred": [], "probs": []}
    }
    
    # 1. Thermostability
    ds = Thermostability(verbose=0); _, _, test = ds.split()
    print("Running Thermostability Inference...")
    for item in tqdm(test):
        seq = " ".join(list(item['sequence']))
        target = item['targets']['target']
        if target is None: continue
        
        inp = engine.backbone.tokenizer([seq], return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            emb = engine.backbone(inp['input_ids'], inp['attention_mask'], 'sequence')
            pred = engine.heads[0](emb).item()
        
        results["Thermo"]["true"].append(target)
        results["Thermo"]["pred"].append(pred)

    # 2. Secondary Structure
    ds = SecondaryStructure(verbose=0); _, _, test = ds.split()
    print("Running SSP Inference...")
    for item in tqdm(test):
        seq = " ".join(list(item['sequence']))
        target = item['targets']['target'] # List
        
        inp = engine.backbone.tokenizer([seq], return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            emb = engine.backbone(inp['input_ids'], inp['attention_mask'], 'token')
            logits = engine.heads[1](emb)
            preds = logits.argmax(dim=-1).squeeze().cpu().tolist()
        
        # Align lengths (remove CLS/SEP)
        valid_len = min(len(target), len(preds)-2)
        p_clean = preds[1 : 1+valid_len]
        t_clean = target[:valid_len]
        
        # Flatten for confusion matrix
        for p, t in zip(p_clean, t_clean):
            if t != -100:
                results["SSP"]["true"].append(t)
                results["SSP"]["pred"].append(p)

    # 3. Cloning
    ds = CloningCLF(verbose=0); _, _, test = ds.split()
    print("Running Cloning Inference...")
    for item in tqdm(test):
        seq = " ".join(list(item['sequence']))
        target = item['targets']['target']
        
        inp = engine.backbone.tokenizer([seq], return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            emb = engine.backbone(inp['input_ids'], inp['attention_mask'], 'sequence')
            logits = engine.heads[2](emb)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            pred = logits.argmax(dim=1).item()
            
        results["Clone"]["true"].append(target)
        results["Clone"]["pred"].append(pred)
        results["Clone"]["probs"].append(prob)

    return results

def plot_diagnostics(results):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- PLOT 1: Thermostability Scatter ---
    t_true = results["Thermo"]["true"]
    t_pred = results["Thermo"]["pred"]
    
    # Calculate R2
    r2 = r2_score(t_true, t_pred)
    
    sns.scatterplot(x=t_true, y=t_pred, alpha=0.5, ax=axes[0], color='blue')
    # Ideal line
    low = min(min(t_true), min(t_pred))
    high = max(max(t_true), max(t_pred))
    axes[0].plot([low, high], [low, high], 'r--', label="Perfect Prediction")
    
    axes[0].set_title(f"Thermostability (MSE=0.21 | R²={r2:.3f})")
    axes[0].set_xlabel("Ground Truth (Z-Score)")
    axes[0].set_ylabel("Predicted")
    axes[0].legend()

    # --- PLOT 2: SSP Confusion Matrix ---
    s_true = results["SSP"]["true"]
    s_pred = results["SSP"]["pred"]
    cm = confusion_matrix(s_true, s_pred, normalize='true') # Normalize by row (True label)
    
    sns.heatmap(cm, annot=False, cmap="Blues", ax=axes[1])
    axes[1].set_title("SecStructure Confusion Matrix\n(Diagonal = Good)")
    axes[1].set_xlabel("Predicted Class (0-7)")
    axes[1].set_ylabel("True Class (0-7)")

    # --- PLOT 3: Cloning Probability Hist ---
    c_true = results["Clone"]["true"]
    c_probs = results["Clone"]["probs"]
    
    df_clone = pd.DataFrame({"Prob": c_probs, "True": c_true})
    sns.histplot(data=df_clone, x="Prob", hue="True", bins=20, ax=axes[2], palette={0:"red", 1:"green"}, element="step")
    
    axes[2].set_title("Cloning Confidence Histogram")
    axes[2].set_xlabel("Predicted Probability of Soluble (1)")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "diagnostic_dashboard.png"), dpi=300)
    print(f"Saved Diagnostic Plot to {SAVE_DIR}")

def save_raw_csv(results):
    # Save the first 100 predictions for manual inspection
    # Thermo
    df_t = pd.DataFrame({"True": results["Thermo"]["true"], "Pred": results["Thermo"]["pred"]})
    df_t.head(100).to_csv(os.path.join(SAVE_DIR, "inspect_thermo.csv"), index=False)
    
    # Clone
    df_c = pd.DataFrame({"True": results["Clone"]["true"], "Pred": results["Clone"]["pred"], "Prob": results["Clone"]["probs"]})
    df_c.head(100).to_csv(os.path.join(SAVE_DIR, "inspect_cloning.csv"), index=False)
    
    print("Saved inspection CSVs (check these manually!)")

if __name__ == "__main__":
    engine = load_model()
    data = gather_test_data(engine)
    save_raw_csv(data)
    plot_diagnostics(data)