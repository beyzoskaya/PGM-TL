import torch
import numpy as np
import pandas as pd
from protbert_hf import SharedProtBert
from engine_hf_cascade import CascadeMultiTaskEngine

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cascade_v1_lora16/model_epoch_5.pt"
WT_SEQ = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

def manual_cascade_forward(engine, inputs_embeds, mask):
    """
    Replicates CascadeMultiTaskEngine.forward but starts from EMBEDDINGS.
    Adjust this if your cascade logic is different!
    Assumed Flow: Backbone -> SSP -> Thermo -> Cloning
    """
    # 1. Backbone
    outputs = engine.backbone.base_model(inputs_embeds=inputs_embeds, attention_mask=mask)
    sequence_output = outputs.last_hidden_state

    # WARNING: If your Cascade passes OUTPUTS of one head as INPUTS to another,
    # you must replicate that specific math here.
    
    # Let's try to inspect the engine structure dynamically or assume simple structure
    # For now, I'll assume independent heads on shared backbone (Parallel) 
    # If it is Serial (Head1 -> Head2), the gradients will naturally flow if connected.

    # SSP (Task 1) - Token level
    ssp_logits = engine.heads[1](sequence_output)
    
    # Pooling for Sequence Tasks
    input_mask_expanded = mask.unsqueeze(-1).expand(sequence_output.size()).float()
    sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    pooled_output = sum_embeddings / sum_mask
    
    # Thermo (Task 0)
    thermo_logits = engine.heads[0](pooled_output)
    
    # Cloning (Task 2)
    cloning_logits = engine.heads[2](pooled_output)
    
    return ssp_logits, thermo_logits, cloning_logits

def get_input_saliency_cascade(engine, tokenizer, seq, target_task_name):
    engine.eval()
    engine.zero_grad()
    
    spaced_seq = " ".join(list(seq))
    inputs = tokenizer([spaced_seq], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(DEVICE)
    input_ids = inputs['input_ids']
    mask = inputs['attention_mask']

    try:
        embed_layer = engine.backbone.base_model.embeddings.word_embeddings
    except:
        embed_layer = engine.backbone.base_model.model.embeddings.word_embeddings
        
    inputs_embeds = embed_layer(input_ids)
    inputs_embeds.retain_grad()
    
    # Run Manual Forward
    ssp, thermo, cloning = manual_cascade_forward(engine, inputs_embeds, mask)
    
    # Select Target
    if target_task_name == 'Thermostability':
        score = thermo
    elif target_task_name == 'Cloning':
        score = cloning.max()
    else:
        score = ssp.max() # Saliency for SSP
        
    score.backward()
    
    grads = inputs_embeds.grad[0] 
    saliency = torch.norm(grads, dim=-1).cpu().numpy()
    return saliency[1:-1]

def main():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    engine = CascadeMultiTaskEngine(backbone, task_configs, [], [], device=DEVICE)
    print(f"Loading Cascade Model: {MODEL_PATH}")
    engine.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
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
    
    df.to_csv("residue_importance_CASCADE.csv", index=False)
    
    corr = np.corrcoef(sal_thermo, sal_cloning)[0,1]
    print(f"\n[Cascade Model] Correlation (Input Grads): {corr:.4f}")

if __name__ == "__main__":
    main()