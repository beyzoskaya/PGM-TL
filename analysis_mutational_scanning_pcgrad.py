import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from protbert_hf import SharedProtBert
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad/model_epoch_5.pt" 

# GFP Sequence
WT_SEQ = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
BATCH_SIZE = 32

def load_pcgrad_model():
    # Initialize Architecture
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    
    engine = MultiTaskEngineHybrid(backbone, task_configs, [], [], device=DEVICE)
    
    print(f"Loading PCGrad weights from {MODEL_PATH}...")
    # FIX: strict=False allows loading even if 'log_vars' is missing
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    engine.load_state_dict(state_dict, strict=False)
    engine.eval()
    return engine, backbone.tokenizer

def predict_batch_pcgrad(engine, tokenizer, seqs):
    spaced_seqs = [" ".join(list(s)) for s in seqs]
    inputs = tokenizer(spaced_seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(DEVICE)
    
    with torch.no_grad():
        # --- Task 0: Thermostability ---
        # Get sequence embedding (pooled)
        emb_thermo = engine.backbone(inputs['input_ids'], inputs['attention_mask'], task_type='sequence')
        out_thermo = engine.heads[0](emb_thermo)

        # --- Task 2: Cloning ---
        emb_cloning = engine.backbone(inputs['input_ids'], inputs['attention_mask'], task_type='sequence')
        out_cloning = engine.heads[2](emb_cloning)
        prob_cloning = torch.softmax(out_cloning, dim=1)[:, 1]
        
    return out_thermo.cpu().numpy().flatten(), prob_cloning.cpu().numpy().flatten()

def main():
    try:
        engine, tokenizer = load_pcgrad_model()
    except Exception as e:
        print(f"\nCRITICAL ERROR loading model: {e}")
        print("Tip: Check if 'unfrozen_layers' in SharedProtBert matches your training (0 vs 2).")
        return

    print(f"Analyzing Sequence (Len: {len(WT_SEQ)})...")
    
    # 1. Wild Type
    wt_thermo, wt_cloning = predict_batch_pcgrad(engine, tokenizer, [WT_SEQ])
    wt_thermo = wt_thermo[0]
    wt_cloning = wt_cloning[0]
    print(f"Wild Type | Thermo Z: {wt_thermo:.4f} | Cloning Prob: {wt_cloning:.4f}")

    # 2. Generate Mutants
    mutants = []
    metadata = [] 
    
    for i, original_aa in enumerate(WT_SEQ):
        for aa in AMINO_ACIDS:
            if aa == original_aa: continue
            mut_seq = WT_SEQ[:i] + aa + WT_SEQ[i+1:]
            mutants.append(mut_seq)
            metadata.append((i+1, original_aa, aa))

    # 3. Batch Inference
    delta_thermo = []
    delta_cloning = []
    
    print(f"Scanning {len(mutants)} variants...")
    for i in tqdm(range(0, len(mutants), BATCH_SIZE)):
        batch_seqs = mutants[i : i+BATCH_SIZE]
        t_preds, c_preds = predict_batch_pcgrad(engine, tokenizer, batch_seqs)
        
        delta_thermo.extend(t_preds - wt_thermo)
        delta_cloning.extend(c_preds - wt_cloning)

    # 4. Save/Plot
    df = pd.DataFrame(metadata, columns=["Position", "WT_AA", "Mut_AA"])
    df["d_Thermo"] = delta_thermo
    df["d_Cloning"] = delta_cloning

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="d_Thermo", y="d_Cloning", alpha=0.5, color='gray')
    
    plt.axhline(0, color='black', linestyle='--'); plt.axvline(0, color='black', linestyle='--')
    plt.title(f"PCGrad Baseline: Mutational Scanning")
    plt.xlabel("Δ Thermostability"); plt.ylabel("Δ Cloning Probability")
    
    plt.tight_layout()
    #plt.savefig("mutational_scanning_PCGRAD.png", dpi=300)
    plt.savefig("/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_all_frozen/mutational_scanning_PCGRAD.png", dpi=300)
    print("Saved plot to mutational_scanning_PCGRAD.png")
    df.to_csv("mutational_scanning_data_PCGRAD.csv", index=False)

if __name__ == "__main__":
    main()