import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from protbert_hf import SharedProtBert
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid
from flip_hf import Thermostability, SecondaryStructure, CloningCLF

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad/model_epoch_5.pt" # PCGrad Weights

KD_SCALE = {
    'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
    'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
    'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
    'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2
}

def calculate_hydrophobicity(seq):
    # Average hydrophobicity (GRAVY score)
    scores = [KD_SCALE.get(aa, 0) for aa in seq]
    return np.mean(scores) if scores else 0

def load_pcgrad_model():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0) 
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    engine = MultiTaskEngineHybrid(backbone, task_configs, [], [], device=DEVICE)
    engine.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    engine.eval()
    return engine

def get_predictions(engine, dataset, task_idx):
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=lambda b: engine.train_loaders[0].collate_fn(b))
    preds, truths, seqs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Task {task_idx}"):
            input_ids = batch['input_ids'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            # Simple retrieval of raw sequences from dataset logic is hard in batch
            # So we just infer properties from input_ids (non-padding)
            for i in range(input_ids.shape[0]):
                # Reconstruct rough seq len
                valid_len = (input_ids[i] != 0).sum().item() - 2 # minus CLS/SEP
                seqs.append(valid_len) # Store Length for now
            
            # Forward
            task_type = engine.task_configs[task_idx]['type']
            emb = engine.backbone(input_ids, batch['attention_mask'].to(DEVICE), 
                                  task_type='token' if task_type=='token_classification' else 'sequence')
            logits = engine.heads[task_idx](emb)
            
            if task_idx == 0: # Thermo
                preds.extend(logits.view(-1).cpu().numpy())
                truths.extend(targets.view(-1).cpu().numpy())
            elif task_idx == 2: # Cloning
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                truths.extend(targets.cpu().numpy())
                
    return np.array(preds), np.array(truths), np.array(seqs)

def main():
    print("Loading Data...")
    # Load Test Sets only
    ds_t = Thermostability(verbose=0); _, _, test_t = ds_t.split()
    ds_c = CloningCLF(verbose=0); _, _, test_c = ds_c.split()
    
    # Calculate Hydrophobicity for Thermo Test Set
    # (Accessing underlying dataset for raw strings)
    raw_seqs_t = [test_t.dataset.sequences[i] for i in test_t.indices]
    hydro_scores = [calculate_hydrophobicity(s) for s in raw_seqs_t]
    
    engine = load_pcgrad_model()
    
    # --- ANALYSIS 1: Thermostability Error vs Hydrophobicity ---
    print("Analyzing Thermo vs Hydrophobicity...")
    p_t, t_t, lens_t = get_predictions(engine, test_t, 0)
    
    # Normalize targets if they aren't already (assuming raw model output matches target scale)
    abs_error = np.abs(p_t - t_t)
    
    df_hydro = pd.DataFrame({'Hydrophobicity': hydro_scores, 'AbsError': abs_error})
    # Binning
    df_hydro['Hydro_Bin'] = pd.cut(df_hydro['Hydrophobicity'], bins=5)
    
    # --- ANALYSIS 2: Cloning Accuracy vs Sequence Length ---
    print("Analyzing Cloning vs Length...")
    p_c, t_c, lens_c = get_predictions(engine, test_c, 2)
    correct = (p_c == t_c).astype(int)
    
    df_len = pd.DataFrame({'Length': lens_c, 'Accuracy': correct})
    # Binning (e.g., <200, 200-400, etc)
    df_len['Len_Bin'] = pd.cut(df_len['Length'], bins=[0, 200, 400, 600, 800, 2000], labels=["<200", "200-400", "400-600", "600-800", ">800"])

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Hydrophobicity Bias
    sns.barplot(data=df_hydro, x='Hydro_Bin', y='AbsError', ax=axes[0], palette="coolwarm", capsize=.1)
    axes[0].set_title("PCGrad Bias: Stability Error vs. Hydrophobicity")
    axes[0].set_ylabel("Mean Absolute Error (MSE)")
    axes[0].set_xlabel("Hydrophobicity (GRAVY Score)\n(Negative = Hydrophilic, Positive = Hydrophobic)")
    
    # Plot 2: Length Bias
    sns.barplot(data=df_len, x='Len_Bin', y='Accuracy', ax=axes[1], palette="viridis", capsize=.1)
    axes[1].set_title("PCGrad Bias: Cloning Accuracy vs. Protein Length")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Sequence Length (Residues)")
    axes[1].axhline(0.76, color='k', linestyle='--', label="Global Avg")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad/pcgrad_biophysical_bias.png")
    print("Saved pcgrad_biophysical_bias.png")

if __name__ == "__main__":
    main()