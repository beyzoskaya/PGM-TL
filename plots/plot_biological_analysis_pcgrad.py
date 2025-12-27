import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engines.engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32 
LORA_RANK = 16
UNFROZEN_LAYERS = 0 

PROJECT_DIR = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad"
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "model_epoch_2.pt") 
STATS_FILE = os.path.join(PROJECT_DIR, "pcgrad_statistics.csv")

def load_model_and_data():
    print("⏳ Loading Data and Model...")
    
    # 1. Data
    ds_thermo = Thermostability(verbose=0); _, _, thermo_test = ds_thermo.split()
    ds_ssp = SecondaryStructure(verbose=0); _, _, ssp_test = ds_ssp.split()
    ds_clf = CloningCLF(verbose=0); _, _, clf_test = ds_clf.split()
    
    # 2. Model
    backbone = SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=UNFROZEN_LAYERS)
    
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    
    engine = MultiTaskEngineHybrid(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=[thermo_test], 
        valid_sets=None,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        engine.load_state_dict(state_dict)
        engine.eval()
        engine.to(DEVICE)
        print("✅ Model Loaded Successfully.")
    else:
        print(f"❌ Weights not found at {CHECKPOINT_PATH}")
        return None, None, None, None

    return engine, thermo_test, ssp_test, clf_test

# ==========================================
# PLOT 1: PCGRAD STATISTICS (Mechanical)
# ==========================================
def plot_pcgrad_dynamics():
    print("\n📊 [Plot 1] Generating PCGrad Conflict Analysis...")
    
    if not os.path.exists(STATS_FILE):
        print("   ❌ CSV not found.")
        return

    df = pd.read_csv(STATS_FILE)
    STEPS_PER_EPOCH = 3351 # Approx
    df['Global_Step'] = (df['Epoch'] - 1) * STEPS_PER_EPOCH + df['Step']
    df = df.drop_duplicates(subset=['Global_Step'], keep='last').sort_values('Global_Step')

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Left Axis: Count
    color = 'tab:red'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Conflicts per Step (Max 6)', color=color)
    ax1.plot(df['Global_Step'], df['Conflict_Count'].rolling(window=20).mean(), 
             color=color, label='Conflict Count')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 7)

    # Right Axis: Magnitude
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Gradient Magnitude Removed', color=color)
    ax2.plot(df['Global_Step'], df['Avg_Conflict_Magnitude'].rolling(window=20).mean(), 
             color=color, linestyle='--', label='Magnitude')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("PCGrad Surgery: Conflict Resolution Dynamics")
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "pcgrad_conflict_dynamics.png"))
    plt.show()
    print(f"   > Avg Conflicts: {df['Conflict_Count'].mean():.2f}")

# ==========================================
# PLOT 2: DMS HEATMAP (Biological)
# ==========================================
def plot_dms(engine, dataset):
    print("\n🔬 [Plot 2] In-Silico Deep Mutational Scanning...")
    
    # Find a LONG protein (200-400 AA) for better visualization
    target_seq = ""
    for i in range(len(dataset)):
        seq = dataset[i]['sequence']
        if 200 < len(seq) < 400:
            target_seq = seq
            print(f"   Found protein at index {i} (Length: {len(seq)})")
            break
    
    if target_seq == "": 
        target_seq = dataset[0]['sequence'] # Fallback
        print(f"   Fallback to short protein (Length: {len(target_seq)})")

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    heatmap_data = np.zeros((20, len(target_seq)))
    
    # Create batch
    batch_seqs = []
    coords = []
    
    for pos_i in range(len(target_seq)):
        for aa_i, aa in enumerate(amino_acids):
            mut = list(target_seq)
            mut[pos_i] = aa
            batch_seqs.append(" ".join(mut))
            coords.append((aa_i, pos_i))
            
    # Predict
    preds = []
    with torch.no_grad():
        for i in range(0, len(batch_seqs), 64):
            batch = batch_seqs[i:i+64]
            inputs = engine.backbone.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(DEVICE)
            emb = engine.backbone(inputs['input_ids'], inputs['attention_mask'], task_type='sequence')
            p = engine.heads[0](emb).flatten().cpu().numpy()
            preds.extend(p)
            
    for p, (r, c) in zip(preds, coords):
        heatmap_data[r, c] = p
        
    plt.figure(figsize=(18, 6))
    sns.heatmap(heatmap_data, yticklabels=amino_acids, cmap="coolwarm", center=0)
    plt.title(f"Stability Landscape (PCGrad Model) | Length: {len(target_seq)}")
    plt.xlabel("Residue Position")
    plt.ylabel("Mutation")
    plt.savefig(os.path.join(PROJECT_DIR, "dms_heatmap.png"))
    plt.show()

# ==========================================
# PLOT 3: SSP CONFUSION MATRIX (Performance)
# ==========================================
def plot_ssp_cm(engine, dataset):
    print("\n🧬 [Plot 3] Secondary Structure Confusion Matrix...")
    
    all_preds = []
    all_targets = []
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, 
                                         collate_fn=lambda b: engine.train_loaders[0].collate_fn(b)) # Use engines collate
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > 20: break 
            
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            # Task 1 = SSP
            emb = engine.backbone(ids, mask, task_type='token')
            logits = engine.heads[1](emb)
            
            p = logits.argmax(dim=-1).view(-1).cpu().numpy()
            t = targets.view(-1).cpu().numpy()
            
            valid = t != -100
            all_preds.extend(p[valid])
            all_targets.extend(t[valid])
            
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
    plt.title("Secondary Structure Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.savefig(os.path.join(PROJECT_DIR, "ssp_confusion_matrix.png"))
    plt.show()

# ==========================================
# PLOT 4: CORRELATION (Validation)
# ==========================================
def plot_correlation(engine, dataset):
    print("\n⚗️ [Plot 4] Stability vs Solubility Correlation...")
    
    t_preds = []
    c_probs = []
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=lambda b: [x['sequence'] for x in b])
    
    with torch.no_grad():
        for i, seqs in enumerate(loader):
            if i > 40: break
            seqs_fmt = [" ".join(list(s)) for s in seqs]
            inputs = engine.backbone.tokenizer(seqs_fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
            emb = engine.backbone(inputs['input_ids'], inputs['attention_mask'], task_type='sequence')
            
            # Head 0 (Thermo), Head 2 (Cloning)
            t_preds.extend(engine.heads[0](emb).flatten().cpu().numpy())
            c_probs.extend(torch.softmax(engine.heads[2](emb), dim=1)[:, 1].cpu().numpy())
            
    corr = np.corrcoef(t_preds, c_probs)[0, 1]
    
    plt.figure(figsize=(6, 6))
    plt.scatter(t_preds, c_probs, alpha=0.5, c=c_probs, cmap='viridis', s=15)
    plt.title(f"Multi-Task Correlation (PCGrad)\nPearson r: {corr:.4f}")
    plt.xlabel("Predicted Stability")
    plt.ylabel("Predicted Solubility Probability")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PROJECT_DIR, "stability_solubility_correlation.png"))
    plt.show()

if __name__ == "__main__":
    engine, thermo, ssp, clf = load_model_and_data()
    
    if engine is not None:
        plot_pcgrad_dynamics()
        plot_dms(engine, thermo)
        plot_ssp_cm(engine, ssp)
        plot_correlation(engine, thermo)