import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from protbert_hf import SharedProtBert
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid
from flip_hf import SecondaryStructure

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad/model_epoch_5.pt"

def main():
    backbone = SharedProtBert(lora_rank=16, unfrozen_layers=0)
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    engine = MultiTaskEngineHybrid(backbone, task_configs, [], [], device=DEVICE)
    engine.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    engine.eval()

    ds = SecondaryStructure(verbose=0); _, _, test = ds.split()
    loader = torch.utils.data.DataLoader(test, batch_size=16, collate_fn=lambda b: engine.train_loaders[0].collate_fn(b))
    
    all_preds = []
    all_truths = []
    
    print("Running Inference on SSP...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(DEVICE)
            targets = batch['targets'].to(DEVICE) # [B, L]
            
            # Forward
            emb = engine.backbone(input_ids, batch['attention_mask'].to(DEVICE), task_type='token')
            logits = engine.heads[1](emb) # Head 1 is SSP
            preds = logits.argmax(dim=-1) # [B, L]
            
            # Flatten and Mask
            mask = targets != -100
            all_preds.extend(preds[mask].cpu().numpy())
            all_truths.extend(targets[mask].cpu().numpy())

    # --- PLOT ---
    # Q8 Labels: 0:G, 1:H, 2:I, 3:E, 4:B, 5:T, 6:S, 7:C (Coil)
    # Grouping: Helix (G,H,I), Strand (E,B), Loop (T,S,C)
    labels = ['G (3-Helix)', 'H (Alpha)', 'I (5-Helix)', 'E (Strand)', 'B (Bridge)', 'T (Turn)', 'S (Bend)', 'C (Coil)']
    
    cm = confusion_matrix(all_truths, all_preds, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("PCGrad Secondary Structure Confusion Matrix (Q8)")
    plt.xlabel("Predicted Structure")
    plt.ylabel("True Structure")
    
    # Draw boxes around groups to show biological clusters
    # Helix (0-2), Strand (3-4), Loop (5-7)
    plt.axhline(3, color='red', lw=2)
    plt.axvline(3, color='red', lw=2)
    plt.axhline(5, color='red', lw=2)
    plt.axvline(5, color='red', lw=2)
    
    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_hybrid_pcgrad/pcgrad_ssp_confusion.png")
    print("Saved pcgrad_ssp_confusion.png")

if __name__ == "__main__":
    main()