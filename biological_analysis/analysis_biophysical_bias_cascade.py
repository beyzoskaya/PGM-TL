import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from protbert_hf import SharedProtBert

from engine_hf_cascade import CascadeMultiTaskEngine, multitask_collate_fn
from flip_hf import Thermostability, CloningCLF

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cascade_v1_lora16/model_epoch_5.pt"

KD_SCALE = {'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5, 'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5, 'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6, 'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2}

def calculate_hydrophobicity(seq):
    scores = [KD_SCALE.get(aa, 0) for aa in seq]
    return np.mean(scores) if scores else 0

def get_predictions_cascade(engine, dataset, target_task_idx):
    tokenizer = engine.backbone.tokenizer
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=lambda b: multitask_collate_fn(b, tokenizer))
    
    preds, truths, seqs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            for i in range(input_ids.shape[0]):
                valid_len = (input_ids[i] != 0).sum().item() - 2
                seqs.append(valid_len)
            
            # Cascade Forward: (ssp, thermo, cloning)
            out_ssp, out_thermo, out_cloning = engine(input_ids, batch['attention_mask'].to(DEVICE))
            
            if target_task_idx == 0: # Thermo
                preds.extend(out_thermo.view(-1).cpu().numpy())
                truths.extend(targets.view(-1).cpu().numpy())
            elif target_task_idx == 2: # Cloning
                preds.extend(out_cloning.argmax(dim=1).cpu().numpy())
                truths.extend(targets.cpu().numpy())
                
    return np.array(preds), np.array(truths), np.array(seqs)

def main():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    engine = CascadeMultiTaskEngine(backbone, task_configs, [], [], device=DEVICE)
    engine.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    engine.eval()
    
    ds_t = Thermostability(verbose=0); _, _, test_t = ds_t.split()
    ds_c = CloningCLF(verbose=0); _, _, test_c = ds_c.split()
    
    raw_seqs_t = [test_t.dataset.sequences[i] for i in test_t.indices]
    hydro_scores = [calculate_hydrophobicity(s) for s in raw_seqs_t]
    
    print("Cascade: Thermo vs Hydrophobicity...")
    p_t, t_t, _ = get_predictions_cascade(engine, test_t, 0)
    abs_error = np.abs(p_t - t_t)
    df_hydro = pd.DataFrame({'Hydrophobicity': hydro_scores, 'AbsError': abs_error})
    df_hydro['Hydro_Bin'] = pd.cut(df_hydro['Hydrophobicity'], bins=5)
    
    print("Cascade: Cloning vs Length...")
    p_c, t_c, lens_c = get_predictions_cascade(engine, test_c, 2)
    correct = (p_c == t_c).astype(int)
    df_len = pd.DataFrame({'Length': lens_c, 'Accuracy': correct})
    df_len['Len_Bin'] = pd.cut(df_len['Length'], bins=[0, 200, 400, 600, 800, 2000], labels=["<200", "200-400", "400-600", "600-800", ">800"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=df_hydro, x='Hydro_Bin', y='AbsError', ax=axes[0], palette="coolwarm", capsize=.1)
    axes[0].set_title("Cascade Bias: Stability Error")
    sns.barplot(data=df_len, x='Len_Bin', y='Accuracy', ax=axes[1], palette="viridis", capsize=.1)
    axes[1].set_title("Cascade Bias: Cloning Accuracy")
    
    plt.tight_layout()
    plt.savefig("cascade_biophysical_bias.png")

if __name__ == "__main__":
    main()