import torch
import numpy as np
import pandas as pd
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from protbert_hf import SharedProtBert
from engines.engine_hf_cascade import CascadeMultiTaskEngine

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cascade_v1_lora16/model_epoch_5.pt" 
WT_SEQ = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

def manual_cascade_forward(engine, inputs_embeds, mask):
  
    outputs = engine.backbone.base_model(inputs_embeds=inputs_embeds, attention_mask=mask)
    base_emb = outputs.last_hidden_state # [Batch, Len, 1024]
    
    ssp_logits = engine.head_ssp(base_emb)
    
    ssp_features = engine.cascade_adapter(ssp_logits)
    
    # Residual Connection (Enriched Embeddings)
    enriched_emb = base_emb + ssp_features
    
    # Pooling (Manually implemented for Enriched Embeddings)
    input_mask_expanded = mask.unsqueeze(-1).expand(enriched_emb.size()).float()
    sum_embeddings = torch.sum(enriched_emb * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    pooled_emb = sum_embeddings / sum_mask
    
    # Functional Predictions
    # Uses self.head_thermo and self.head_cloning
    thermo_pred = engine.head_thermo(pooled_emb)
    cloning_pred = engine.head_cloning(pooled_emb)

    return ssp_logits, thermo_pred, cloning_pred

def get_input_saliency_cascade(engine, tokenizer, seq, target_task_name):
    engine.eval()
    engine.zero_grad()
    
    spaced_seq = " ".join(list(seq))
    inputs = tokenizer([spaced_seq], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(DEVICE)
    input_ids = inputs['input_ids']
    mask = inputs['attention_mask']

    # Hook into Embeddings
    try:
        embed_layer = engine.backbone.base_model.embeddings.word_embeddings
    except:
        embed_layer = engine.backbone.base_model.model.embeddings.word_embeddings
        
    inputs_embeds = embed_layer(input_ids)
    inputs_embeds.retain_grad()

    ssp, thermo, cloning = manual_cascade_forward(engine, inputs_embeds, mask)
    
    if target_task_name == 'Thermostability':
        score = thermo
    elif target_task_name == 'Cloning':
        # Cloning is [Batch, 2], we want the max class score
        score = cloning.max()
    else:
        # SSP is [Batch, Len, 8], max score
        score = ssp.max() 
        
    score.backward()
    
    # Extract Gradients
    grads = inputs_embeds.grad[0] 
    saliency = torch.norm(grads, dim=-1).cpu().numpy()
    
    # Remove CLS/SEP
    return saliency[1:-1]

def main():
    # Initialize Architecture
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    
    # Initialize Engine (Empty data loaders)
    engine = CascadeMultiTaskEngine(backbone, task_configs, [], [], device=DEVICE)
    
    print(f"Loading Cascade Model: {MODEL_PATH}")
    try:
        engine.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    tokenizer = backbone.tokenizer
    
    print("Calculating Input-Gradient Saliency (Cascade)...")
    
    sal_thermo = get_input_saliency_cascade(engine, tokenizer, WT_SEQ, 'Thermostability')
    sal_cloning = get_input_saliency_cascade(engine, tokenizer, WT_SEQ, 'Cloning')
    
    # Normalize
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
    
    output_path = "/content/drive/MyDrive/protein_multitask_outputs/cascade_v1_lora16/residue_importance_CASCADE.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    # Correlation Analysis
    corr = np.corrcoef(sal_thermo, sal_cloning)[0,1]
    print(f"\n[Cascade Model] Correlation (Input Grads): {corr:.4f}")
    
    print("-" * 30)
    print("Interpretation:")
    print("In Cascade, HIGH correlation (>0.7) is expected and NOT BAD.")
    print("Why? Because Thermo and Cloning both depend on the SAME Enriched Embedding.")
    print("This contrasts with PCGrad (Entangled=Bad) vs Prompted (Separated=Good).")

if __name__ == "__main__":
    main()

# [Cascade Model] Correlation (Input Grads): 0.2575