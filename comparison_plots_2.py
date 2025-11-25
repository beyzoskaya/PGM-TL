import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

from engine_hf_with_task_specific_encoder import MultiTaskEngine as StandardEngine
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid as HybridEngine
from flip_hf import SecondaryStructure
from protbert_hf import SharedProtBert

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = "/content/drive/MyDrive/protein_multitask_outputs"
SAMPLE_INDEX = 250
SEQ_LEN_LIMIT = 60 

PATH_BASELINE = os.path.join(BASE_DIR, "baseline_SecStructure_lora_only/model_epoch_5.pt")
PATH_PCGRAD = os.path.join(BASE_DIR, "cyclic_v1_lora16_hybrid_pcgrad/model_epoch_2.pt")

# --- Q8 MAPPING & COLORS ---
# Q8 Label mapping (Standard for SSP)
# 0: G (3-10 Helix) | 1: H (Alpha Helix) | 2: I (Pi Helix) -> REDS
# 3: T (Turn) | 4: E (Beta Sheet) | 5: B (Bridge) -> BLUES/YELLOWS
# 6: S (Bend) | 7: C (Coil) -> GRAYS
ID2LABEL = {
    0: "G (3-10 Helix)", 1: "H (Alpha Helix)", 2: "I (Pi Helix)",
    3: "T (Turn)", 4: "E (Beta Sheet)", 5: "B (Bridge)",
    6: "S (Bend)", 7: "C (Coil)"
}

# Helix = Reds/Oranges
# Sheet = Blues/Cyans
# Coil/Turn = Grays/Whites
COLOR_MAP = {
    0: "#ff9999", 1: "#cc0000", 2: "#ff3333", # Helices (Red)
    3: "#ffff99", # Turn (Yellow)
    4: "#0000cc", 5: "#66b2ff", # Sheets (Blue)
    6: "#cccccc", 7: "#ffffff"  # Coils (Gray/White)
}

def load_models():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    
    # Load Baseline
    cfg_base = [{'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8}]
    m_base = StandardEngine(backbone, cfg_base, [], [], batch_size=1, device=DEVICE)
    if os.path.exists(PATH_BASELINE):
        m_base.load_state_dict(torch.load(PATH_BASELINE, map_location=DEVICE))
    m_base.eval().to(DEVICE)

    # Load PCGrad
    cfg_mtl = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    m_pc = HybridEngine(backbone, cfg_mtl, [], [], batch_size=1, device=DEVICE)
    if os.path.exists(PATH_PCGRAD):
        m_pc.load_state_dict(torch.load(PATH_PCGRAD, map_location=DEVICE), strict=False)
    m_pc.eval().to(DEVICE)
    
    return m_base, m_pc

def predict_sequence(model, sequence, model_type="Baseline"):
    tokenizer = model.backbone.tokenizer
    inp = tokenizer([sequence], return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
    
    with torch.no_grad():
        emb = model.backbone(inp['input_ids'], inp['attention_mask'], task_type='token')
        
        # Select correct head
        if model_type == "Baseline":
            logits = model.heads[0](emb)
        else: # PCGrad (SecStruct is usually index 1 in your config)
            logits = model.heads[1](emb)
            
        preds = logits.argmax(dim=-1).squeeze().cpu().numpy()
        
    # Remove CLS (index 0) and SEP (last) tokens
    return preds[1 : len(sequence)+1]

def plot_structure_track(aa_seq, true_seq, base_seq, pc_seq, index):
    """
    Creates the alignment plot.
    """
    # Create Matrix: Rows=Models, Cols=Residues
    # We map labels (0-7) to a color matrix later
    
    # Filter for valid length
    L = min(len(aa_seq), len(true_seq), SEQ_LEN_LIMIT)
    aa_seq = aa_seq[:L]
    
    # Prepare data for heatmap
    data_matrix = np.array([
        true_seq[:L],
        base_seq[:L],
        pc_seq[:L]
    ])
    
    fig, ax = plt.subplots(figsize=(L * 0.3, 4)) # Adjust width based on length
    
    # Custom Color Map for Matplotlib
    # We create a ListedColormap based on our Q8 keys 0..7
    from matplotlib.colors import ListedColormap
    c_list = [COLOR_MAP[i] for i in range(8)]
    cmap = ListedColormap(c_list)
    
    # Plot Heatmap
    sns.heatmap(data_matrix, cmap=cmap, cbar=False, ax=ax, linewidths=0.5, linecolor='black', vmin=0, vmax=7)
    
    # Labels
    ax.set_yticklabels(["Ground Truth", "Baseline Pred", "PCGrad Pred"], rotation=0, fontweight='bold')
    ax.set_xticklabels(list(aa_seq), rotation=0, fontsize=10)
    ax.set_xlabel("Amino Acid Sequence")
    ax.set_title(f"Protein Sample #{index}: Secondary Structure Prediction Alignment", fontsize=14)
    
    # Create Legend Manually
    patches = [mpatches.Patch(color=COLOR_MAP[i], label=ID2LABEL[i]) for i in [1, 4, 7, 0, 3]] # Show main ones
    plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', title="Structure Type")
    
    plt.tight_layout()
    plt.savefig(f"/content/drive/MyDrive/protein_multitask_outputs/final_plots/SSP_Track_Sample_{index}_epoch_2_weights.png", dpi=150)
    plt.show()

def main():
    print("Loading Data...")
    ds = SecondaryStructure(verbose=0)
    _, _, test_set = ds.split()
    
    print(f"Loading Models... (Analyzing Sample {SAMPLE_INDEX})")
    m_base, m_pc = load_models()
    
    # Get Data
    item = test_set[SAMPLE_INDEX]
    raw_seq = item['sequence']
    fmt_seq = " ".join(list(raw_seq)) # ProtBert format
    targets = item['targets']['target']
    
    # Get Predictions
    p_base = predict_sequence(m_base, fmt_seq, "Baseline")
    p_pc = predict_sequence(m_pc, fmt_seq, "PCGrad")
    
    # Handle Target Padding (-100)
    # Convert -100 in targets to 7 (Coil) or similar for plotting visualization
    clean_targets = [t if t != -100 else 7 for t in targets]
    
    print("Generating Plot...")
    plot_structure_track(raw_seq, clean_targets, p_base, p_pc, SAMPLE_INDEX)
    print(f"Done! Check SSP_Track_Sample_{SAMPLE_INDEX}.png")

if __name__ == "__main__":
    main()