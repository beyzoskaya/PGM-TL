import torch
import numpy as np
import pandas as pd
from protbert_hf import SharedProtBert
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad/model_epoch_5.pt" 
WT_SEQ = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

def manual_mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_saliency_pcgrad(engine, tokenizer, seq, task_idx):
    engine.eval()
    engine.zero_grad()
    
    spaced_seq = " ".join(list(seq))
    inputs = tokenizer([spaced_seq], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(DEVICE)
    input_ids = inputs['input_ids']
    mask = inputs['attention_mask']

    # 1. Get Token Embeddings (Force task_type='token' to get gradients on residues)
    token_out = engine.backbone(input_ids, mask, task_type='token')
    
    # 2. Hook for gradients
    embeddings = token_out.clone().detach().requires_grad_(True)
    
    # 3. Prepare input for Head
    task_type = engine.task_configs[task_idx]['type']
    
    if task_type in ['regression', 'sequence_classification']:
        final_input = manual_mean_pooling(embeddings, mask)
    else:
        # For SSP (task 1)
        final_input = embeddings

    # 4. Forward Head
    logits = engine.heads[task_idx](final_input)
    
    # 5. Backward
    if logits.shape[-1] == 1: 
        score = logits
    else: 
        score = logits.max()
        
    score.backward()
    
    # 6. Extract Gradients
    grads = embeddings.grad[0] # [SeqLen, Hidden]
    saliency = torch.norm(grads, dim=-1).cpu().numpy()
    
    return saliency[1:-1]

def main():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0) # Adjust unfrozen if needed
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    engine = MultiTaskEngineHybrid(backbone, task_configs, [], [], device=DEVICE)
    
    print(f"Loading PCGrad Model: {MODEL_PATH}")
    # FIX: strict=False
    engine.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    tokenizer = backbone.tokenizer
    
    print("Calculating Saliency Maps (PCGrad)...")
    
    # Thermo Saliency
    sal_thermo = get_saliency_pcgrad(engine, tokenizer, WT_SEQ, task_idx=0)
    
    # Cloning Saliency
    sal_cloning = get_saliency_pcgrad(engine, tokenizer, WT_SEQ, task_idx=2)
    
    # Normalize
    sal_thermo = (sal_thermo - sal_thermo.min()) / (sal_thermo.max() - sal_thermo.min())
    sal_cloning = (sal_cloning - sal_cloning.min()) / (sal_cloning.max() - sal_cloning.min())
    
    # Save
    df = pd.DataFrame({
        "Residue": list(WT_SEQ),
        "Position": range(1, len(WT_SEQ)+1),
        "Importance_Thermo": sal_thermo,
        "Importance_Cloning": sal_cloning
    })
    
    #df.to_csv("residue_importance_PCGRAD.csv", index=False)
    df.to_csv("/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_all_frozen/residue_importance_PCGRAD.csv", index=False)
    print("Saved: residue_importance_PCGRAD.csv")
    
    correlation = np.corrcoef(sal_thermo, sal_cloning)[0,1]
    print(f"\n[PCGrad Baseline] Correlation between Thermo/Cloning Attention: {correlation:.4f}")

if __name__ == "__main__":
    main()