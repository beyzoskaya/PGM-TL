import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from protbert_hf import SharedProtBert
from engine_hf_cascade import CascadeMultiTaskEngine

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cascade_v1_lora16/model_epoch_5.pt" 
WT_SEQ = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
BATCH_SIZE = 32

def load_cascade_model():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    engine = CascadeMultiTaskEngine(
        backbone, task_configs, [], [], device=DEVICE
    )
    print(f"Loading Cascade weights from {MODEL_PATH}...")
    engine.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    engine.eval()
    return engine, backbone.tokenizer

def predict_batch_cascade(engine, tokenizer, seqs):
    spaced_seqs = [" ".join(list(s)) for s in seqs]
    inputs = tokenizer(spaced_seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(DEVICE)
    
    with torch.no_grad():
        # Cascade Forward returns tuple: (ssp, thermo, cloning)
        # Verify this order in your engine_hf_cascade.py!
        ssp_out, thermo_out, cloning_out = engine(inputs['input_ids'], inputs['attention_mask'])
        
        # Thermo is Regression (Index 1)
        # Cloning is Classification (Index 2) -> Get Prob of Class 1
        prob_cloning = torch.softmax(cloning_out, dim=1)[:, 1]
        
    return thermo_out.cpu().numpy().flatten(), prob_cloning.cpu().numpy().flatten()

def main():
    engine, tokenizer = load_cascade_model()
    
    print(f"Analyzing Wild Type...")
    wt_thermo, wt_cloning = predict_batch_cascade(engine, tokenizer, [WT_SEQ])
    wt_thermo = wt_thermo[0]
    wt_cloning = wt_cloning[0]
    print(f"Wild Type | Thermo Z: {wt_thermo:.4f} | Cloning Prob: {wt_cloning:.4f}")

    mutants = []
    metadata = [] 
    for i, original_aa in enumerate(WT_SEQ):
        for aa in AMINO_ACIDS:
            if aa == original_aa: continue
            mut_seq = WT_SEQ[:i] + aa + WT_SEQ[i+1:]
            mutants.append(mut_seq)
            metadata.append((i+1, original_aa, aa))

    delta_thermo, delta_cloning = [], []
    
    print(f"Scanning {len(mutants)} variants (Cascade)...")
    for i in tqdm(range(0, len(mutants), BATCH_SIZE)):
        batch_seqs = mutants[i : i+BATCH_SIZE]
        t_preds, c_preds = predict_batch_cascade(engine, tokenizer, batch_seqs)
        delta_thermo.extend(t_preds - wt_thermo)
        delta_cloning.extend(c_preds - wt_cloning)

    df = pd.DataFrame(metadata, columns=["Position", "WT_AA", "Mut_AA"])
    df["d_Thermo"] = delta_thermo
    df["d_Cloning"] = delta_cloning

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="d_Thermo", y="d_Cloning", alpha=0.5, color='teal') # Teal for Cascade
    plt.axhline(0, color='black', linestyle='--'); plt.axvline(0, color='black', linestyle='--')
    plt.title(f"Cascade Model: Mutational Scanning")
    plt.xlabel("Δ Thermostability"); plt.ylabel("Δ Cloning Probability")
    plt.tight_layout()
    plt.savefig("mutational_scanning_CASCADE.png", dpi=300)
    print("Saved plot to mutational_scanning_CASCADE.png")
    df.to_csv("mutational_scanning_data_CASCADE.csv", index=False)

if __name__ == "__main__":
    main()