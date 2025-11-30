import torch
import numpy as np
import pandas as pd
from protbert_hf import SharedProtBert
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad/model_epoch_5.pt"
WT_SEQ = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

def get_input_saliency(engine, tokenizer, seq, task_idx):
    engine.eval()
    engine.zero_grad()

    spaced_seq = " ".join(list(seq))
    inputs = tokenizer([spaced_seq], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(DEVICE)
    input_ids = inputs['input_ids']
    mask = inputs['attention_mask']

    # 2. HACK: Get Embeddings directly from the base HF model
    # This bypasses the tokenizer to allow gradient tracking on "continuous" inputs
    # Access pattern: backbone -> base_model -> model -> embeddings
    # Note: Depending on HF version, it might be backbone.base_model.embeddings
    try:
        embed_layer = engine.backbone.base_model.embeddings.word_embeddings
    except:
        embed_layer = engine.backbone.base_model.model.embeddings.word_embeddings
        
    inputs_embeds = embed_layer(input_ids)
    inputs_embeds.retain_grad() # Crucial: Keep grad for this tensor
    
    # 3. Forward Pass using inputs_embeds instead of input_ids
    # We call the internal model directly to bypass SharedProtBert's input_ids logic
    outputs = engine.backbone.base_model(inputs_embeds=inputs_embeds, attention_mask=mask)
    last_hidden_state = outputs.last_hidden_state
    
    # 4. Manual Pooling (Same as before)
    input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    pooled = sum_embeddings / sum_mask
    
    # 5. Forward Head
    # Handle the head input type
    task_type = engine.task_configs[task_idx]['type']
    if task_type in ['regression', 'sequence_classification']:
        final_input = pooled
    else:
        final_input = last_hidden_state

    logits = engine.heads[task_idx](final_input)
    
    # 6. Backward
    if logits.shape[-1] == 1:
        score = logits
    else:
        score = logits.max()
        
    score.backward()
    
    # 7. Get Gradient w.r.t Input Embeddings
    # This will be [1, SeqLen, Hidden]
    grads = inputs_embeds.grad[0] 
    
    # Compute L2 norm across hidden dimension -> [SeqLen]
    saliency = torch.norm(grads, dim=-1).cpu().numpy()
    
    # Slice [1:-1] to remove CLS/SEP
    return saliency[1:-1]

def main():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    engine = MultiTaskEngineHybrid(backbone, task_configs, [], [], device=DEVICE)
    
    print(f"Loading PCGrad Model: {MODEL_PATH}")
    engine.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    tokenizer = backbone.tokenizer
    
    print("Calculating Input-Gradient Saliency (Fixed)...")
    
    # Thermo
    sal_thermo = get_input_saliency(engine, tokenizer, WT_SEQ, task_idx=0)
    
    # Cloning
    sal_cloning = get_input_saliency(engine, tokenizer, WT_SEQ, task_idx=2)
    
    # Robust Normalization (Avoids divide by zero)
    def normalize(arr):
        denom = arr.max() - arr.min()
        if denom == 0: return np.zeros_like(arr)
        return (arr - arr.min()) / denom

    sal_thermo = normalize(sal_thermo)
    sal_cloning = normalize(sal_cloning)
    
    df = pd.DataFrame({
        "Residue": list(WT_SEQ),
        "Position": range(1, len(WT_SEQ)+1),
        "Importance_Thermo": sal_thermo,
        "Importance_Cloning": sal_cloning
    })
    
    df.to_csv("/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad/residue_importance_PCGRAD_fixed.csv", index=False)
    
    corr = np.corrcoef(sal_thermo, sal_cloning)[0,1]
    print(f"\n[PCGrad Baseline] Correlation (Input Grads): {corr:.4f}")
    
    if corr > 0.7:
        print(">> High correlation! Confirms entanglement (Tasks look at same residues).")
    else:
        print(">> Low correlation. PCGrad separated the attention.")

if __name__ == "__main__":
    main()

# [PCGrad Baseline] Correlation (Input Grads): 0.5700
# >> Low correlation. PCGrad separated the attention.