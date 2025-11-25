import torch
import numpy as np
import matplotlib.pyplot as plt
import biotite.sequence as seq
import biotite.sequence.graphics as graphics
import os

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

# --- MAPPING HELPERS ---
# We map Q8 (8 states) to Q3 (3 biological states) for clear plotting
# 0(G), 1(H), 2(I) -> 'a' (Alpha Helix - displayed as squiggle)
# 4(E), 5(B)       -> 'b' (Beta Sheet - displayed as arrow)
# Others           -> 'c' (Coil - displayed as straight line)
def q8_to_secondary_shape(label_int):
    if label_int in [0, 1, 2]: return 'a' # Helix
    if label_int in [4, 5]:    return 'b' # Sheet
    return 'c' # Coil

def load_models():
    # Reuse your existing loading logic
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
        idx = 0 if model_type == "Baseline" else 1 # Index 1 is SecStruct in PCGrad config
        logits = model.heads[idx](emb)
        preds = logits.argmax(dim=-1).squeeze().cpu().numpy()
    return preds[1:len(sequence)+1] # Remove CLS

def plot_cartoon_comparison(aa_seq, true_q8, base_q8, pc_q8, index):
    # 1. Truncate for visibility
    L = min(len(aa_seq), len(true_q8), SEQ_LEN_LIMIT)
    aa_seq = aa_seq[:L]
    
    # 2. Convert Integers to Biotite Annotation Array
    # 'a' = helix (green squiggle), 'b' = sheet (red arrow), 'c' = coil
    true_ss = [q8_to_secondary_shape(x) for x in true_q8[:L]]
    base_ss = [q8_to_secondary_shape(x) for x in base_q8[:L]]
    pc_ss   = [q8_to_secondary_shape(x) for x in pc_q8[:L]]
    
    # Create Biotite Annotation objects
    annot_true = seq.Annotation(true_ss)
    annot_base = seq.Annotation(base_ss)
    annot_pc   = seq.Annotation(pc_ss)

    # 3. Setup Plot
    fig = plt.figure(figsize=(10, 6))
    
    # Define Subplots (3 tracks)
    # The 'loc' and 'loc_start' align them horizontally
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    # 4. Draw Tracks
    # Ground Truth
    graphics.plot_secondary_structure(
        axes=ax1, 
        feature=annot_true, 
        loc=1, # Start index
        symbols=SEQ_LEN_LIMIT # How many symbols to show per line
    )
    ax1.set_title(f"Ground Truth (Protein #{index})", loc='left', fontsize=12, fontweight='bold')
    ax1.axis('off') # Hide box

    # Baseline
    graphics.plot_secondary_structure(
        axes=ax2, 
        feature=annot_base, 
        loc=1, 
        symbols=SEQ_LEN_LIMIT
    )
    ax2.set_title("Baseline Prediction (Single Task)", loc='left', fontsize=12, color='gray')
    ax2.axis('off')

    # PCGrad
    graphics.plot_secondary_structure(
        axes=ax3, 
        feature=annot_pc, 
        loc=1, 
        symbols=SEQ_LEN_LIMIT
    )
    ax3.set_title("PCGrad Prediction (Multi-Task)", loc='left', fontsize=12, color='darkblue')
    ax3.axis('off')
    
    # 5. Add Legend Manually
    # Biotite uses standard colors: Green=Helix, Red=Sheet, Black=Coil
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Helix (Spiral)'),
        Line2D([0], [0], color='red', lw=2, label='Sheet (Arrow)'),
        Line2D([0], [0], color='black', lw=1, label='Coil (Line)')
    ]
    fig.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    save_path = f"/content/drive/MyDrive/protein_multitask_outputs/final_plots/Bio_Cartoon_Structure_{index}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Comparison saved to {save_path}")
    plt.show()

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

    # Predict
    p_base = get_predictions(m_base, raw_seq, "Baseline")
    p_pc   = get_predictions(m_pc, raw_seq, "PCGrad")
    
    # Plot
    plot_cartoon_comparison(raw_seq, clean_targets, p_base, p_pc, SAMPLE_INDEX)

if __name__ == "__main__":
    main()