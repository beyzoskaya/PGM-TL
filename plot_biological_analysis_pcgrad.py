import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32 
LORA_RANK = 16
UNFROZEN_LAYERS = 0 

CHECKPOINT_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad/model_epoch_2.pt" 

def analyze_pcgrad_results():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Path not found: {CHECKPOINT_PATH}")
        print("Please update CHECKPOINT_PATH to point to your new PCGrad model.")
        return

    print("⏳ Loading PCGrad Model...")
    
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
    
    # Use Hybrid Engine to match keys
    engine = MultiTaskEngineHybrid(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=[thermo_test], 
        valid_sets=None,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    # 3. Load Weights
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    engine.load_state_dict(state_dict)
    engine.eval()
    engine.to(DEVICE)
    print("✅ Model Loaded.")

    # --- ANALYSIS 1: DMS (Longer Protein) ---
    print("\n🔬 [Analysis 1] In-Silico Deep Mutational Scanning")
    
    # Find a good sized protein
    target_seq = ""
    for item in thermo_test:
        if 200 < len(item['sequence']) < 300:
            target_seq = item['sequence']
            break
    
    if target_seq == "": target_seq = thermo_test[0]['sequence'] # Fallback
    
    print(f"   Selected Protein Length: {len(target_seq)} AA")
    
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    heatmap_data = np.zeros((20, len(target_seq)))
    batch_seqs = []
    coords = []
    
    for pos_i, orig_aa in enumerate(target_seq):
        for aa_i, mut_aa in enumerate(amino_acids):
            mut_list = list(target_seq)
            mut_list[pos_i] = mut_aa
            batch_seqs.append(" ".join(mut_list))
            coords.append((aa_i, pos_i))
            
    # Inference
    preds = []
    with torch.no_grad():
        for i in range(0, len(batch_seqs), 64): # Batch 64
            batch = batch_seqs[i:i+64]
            inputs = engine.backbone.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(DEVICE)
            emb = engine.backbone(inputs['input_ids'], inputs['attention_mask'], task_type='sequence')
            p = engine.heads[0](emb).flatten().cpu().numpy()
            preds.extend(p)
            
    for p, (r, c) in zip(preds, coords):
        heatmap_data[r, c] = p
        
    plt.figure(figsize=(20, 6))
    sns.heatmap(heatmap_data, yticklabels=amino_acids, cmap="coolwarm", center=0)
    plt.title(f"PCGrad Stability Landscape (Len={len(target_seq)})")
    plt.xlabel("Position")
    plt.ylabel("Mutation")
    plt.savefig("/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad/pcgrad_dms_heatmap.png", dpi=300)
    plt.show()

    # --- ANALYSIS 2: CORRELATION ---
    print("\n⚗️ [Analysis 2] Stability vs Solubility (PCGrad Check)")
    t_preds = []
    c_probs = []
    
    loader = torch.utils.data.DataLoader(thermo_test, batch_size=32, collate_fn=lambda b: [x['sequence'] for x in b])
    
    with torch.no_grad():
        for i, seqs_raw in enumerate(loader):
            if i > 40: break
            seqs_fmt = [" ".join(list(s)) for s in seqs_raw]
            inputs = engine.backbone.tokenizer(seqs_fmt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
            
            emb = engine.backbone(inputs['input_ids'], inputs['attention_mask'], task_type='sequence')
            
            # Thermo (Head 0)
            t = engine.heads[0](emb).flatten().cpu().numpy()
            t_preds.extend(t)
            
            # Cloning (Head 2)
            c = torch.softmax(engine.heads[2](emb), dim=1)[:, 1].cpu().numpy()
            c_probs.extend(c)
            
    corr = np.corrcoef(t_preds, c_probs)[0, 1]
    print(f"   > PCGrad Correlation: {corr:.4f}")
    
    plt.figure(figsize=(6, 6))
    plt.scatter(t_preds, c_probs, alpha=0.5, c=c_probs, cmap='viridis')
    plt.title(f"PCGrad: Stability vs Solubility (Corr={corr:.2f})")
    plt.xlabel("Stability (Z-Score)")
    plt.ylabel("Solubility Prob")
    plt.grid(True, alpha=0.3)
    plt.savefig("/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad/pcgrad_stability_vs_solubility.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    analyze_pcgrad_results()