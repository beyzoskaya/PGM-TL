import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arrow
import os
from itertools import groupby

from engine_hf_with_task_specific_encoder import MultiTaskEngine as StandardEngine
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid as HybridEngine
from flip_hf import SecondaryStructure
from protbert_hf import SharedProtBert

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = "/content/drive/MyDrive/protein_multitask_outputs"
SAMPLE_INDEX = 100
SEQ_LEN_LIMIT = 80 

PATH_BASELINE = os.path.join(BASE_DIR, "baseline_SecStructure_lora_only/model_epoch_5.pt")
PATH_PCGRAD = os.path.join(BASE_DIR, "cyclic_v1_lora16_hybrid_pcgrad/model_epoch_4.pt")

def get_segments(seq_q8):
    """
    Converts a list of Q8 integers into segments of (Type, Start, Length).
    Mapping:
    - Helix (G, H, I -> 0,1,2): 'Helix'
    - Sheet (E, B -> 4,5):      'Sheet'
    - Coil (Others):            'Coil'
    """
    # 1. Map Q8 to Q3
    q3_seq = []
    for s in seq_q8:
        if s in [0, 1, 2]: q3_seq.append('Helix')
        elif s in [4, 5]:  q3_seq.append('Sheet')
        else:              q3_seq.append('Coil')
    
    # 2. Run-Length Encoding (Group consecutive)
    segments = []
    curr_idx = 0
    for key, group in groupby(q3_seq):
        length = len(list(group))
        segments.append((key, curr_idx, length))
        curr_idx += length
    return segments

def draw_track(ax, segments, length, title, color_scheme):
    """
    Draws a single secondary structure track using Matplotlib Patches.
    """
    # Draw central "Coil" line first (background wire)
    ax.plot([0, length], [0.5, 0.5], color='black', linewidth=1, zorder=0)
    
    for (ss_type, start, duration) in segments:
        if start >= length: break
        
        # Calculate width (clip if it goes over limit)
        width = min(duration, length - start)
        
        if ss_type == 'Helix':
            # Draw Cylinder (FancyBox)
            # x, y, width, height
            p = FancyBboxPatch(
                (start, 0.2), width, 0.6,
                boxstyle="round,pad=0.0,rounding_size=0.3",
                ec="black", fc=color_scheme['Helix'],
                mutation_scale=1, zorder=2
            )
            ax.add_patch(p)
            
        elif ss_type == 'Sheet':
            # Draw Arrow
            # x, y, dx, dy
            # We use mpatches.FancyArrow to look like a biological arrow
            p = mpatches.FancyArrow(
                x=start, y=0.5, dx=width, dy=0,
                width=0.4, length_includes_head=True, 
                head_width=0.8, head_length=min(width, 1.5),
                ec="black", fc=color_scheme['Sheet'], zorder=2
            )
            ax.add_patch(p)
            
    # Styling
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, length + 1)
    ax.set_ylim(0, 1)
    ax.axis('off') # Hide axes

def plot_cartoon_comparison(aa_seq, true_q8, base_q8, pc_q8, index):
    L = min(len(aa_seq), len(true_q8), SEQ_LEN_LIMIT)
    
    # Process Segments
    seg_true = get_segments(true_q8[:L])
    seg_base = get_segments(base_q8[:L])
    seg_pc   = get_segments(pc_q8[:L])
    
    # Colors (Standard Bio Colors)
    colors = {'Helix': '#2ca02c', 'Sheet': '#d62728', 'Coil': 'black'} # Green Helix, Red Sheet
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 5))
    
    draw_track(axes[0], seg_true, L, f"Ground Truth (Protein #{index})", colors)
    draw_track(axes[1], seg_base, L, "Baseline Prediction (Single Task)", colors)
    draw_track(axes[2], seg_pc,   L, "PCGrad Prediction (Multi-Task)", colors)
    
    # Add Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['Helix'], edgecolor='black', label='Helix (Alpha)'),
        mpatches.Patch(facecolor=colors['Sheet'], edgecolor='black', label='Sheet (Beta)'),
        torch.nn.Identity() # Spacer
    ]
    # Draw simple line for Coil legend
    from matplotlib.lines import Line2D
    coil_line = Line2D([0], [0], color='black', lw=1, label='Coil / Loop')
    
    fig.legend(handles=[legend_elements[0], legend_elements[1], coil_line], 
               loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    plt.tight_layout()
    save_name = f"/content/drive/MyDrive/protein_multitask_outputs/final_plots/Bio_Cartoon_Structure_{index}.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ Success! Plot saved to: {save_name}")
    plt.show()

# --- HELPER FUNCTIONS (Same as before) ---
def load_models():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    
    # Baseline
    cfg_base = [{'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8}]
    m_base = StandardEngine(backbone, cfg_base, [], [], batch_size=1, device=DEVICE)
    if os.path.exists(PATH_BASELINE):
        m_base.load_state_dict(torch.load(PATH_BASELINE, map_location=DEVICE))
    m_base.eval().to(DEVICE)

    # PCGrad
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

def get_predictions(model, sequence, model_type):
    tokenizer = model.backbone.tokenizer
    inp = tokenizer([sequence], return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        emb = model.backbone(inp['input_ids'], inp['attention_mask'], task_type='token')
        idx = 0 if model_type == "Baseline" else 1 
        logits = model.heads[idx](emb)
        preds = logits.argmax(dim=-1).squeeze().cpu().numpy()
    return preds[1:len(sequence)+1]

def main():
    print("Loading Data...")
    ds = SecondaryStructure(verbose=0)
    _, _, test_set = ds.split()
    
    print("Loading Models...")
    m_base, m_pc = load_models()
    
    # Get Data
    item = test_set[SAMPLE_INDEX]
    raw_seq = item['sequence']
    targets = item['targets']['target']
    clean_targets = [t if t != -100 else 7 for t in targets]

    print("Running Inference...")
    p_base = get_predictions(m_base, raw_seq, "Baseline")
    p_pc   = get_predictions(m_pc, raw_seq, "PCGrad")
    
    print("Plotting Cartoon...")
    plot_cartoon_comparison(raw_seq, clean_targets, p_base, p_pc, SAMPLE_INDEX)

if __name__ == "__main__":
    main()