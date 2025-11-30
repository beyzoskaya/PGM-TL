import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from protbert_hf import SharedProtBert
from engine_hf_cascade import CascadeMultiTaskEngine, multitask_collate_fn
from flip_hf import SecondaryStructure

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cascade_v1_lora16/model_epoch_5.pt"

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
    
    ds = SecondaryStructure(verbose=0); _, _, test = ds.split()
    loader = torch.utils.data.DataLoader(test, batch_size=16, collate_fn=lambda b: multitask_collate_fn(b, engine.backbone.tokenizer))
    
    all_preds, all_truths = [], []
    
    print("Cascade Inference on SSP...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            # Unpack Tuple. SSP is usually Index 0 or 1. 
            # In eval script: ssp_p, thermo_p, clone_p = engine(...)
            out_ssp, _, _ = engine(input_ids, batch['attention_mask'].to(DEVICE))
            
            preds = out_ssp.argmax(dim=-1)
            mask = targets != -100
            all_preds.extend(preds[mask].cpu().numpy())
            all_truths.extend(targets[mask].cpu().numpy())

    labels = ['G', 'H', 'I', 'E', 'B', 'T', 'S', 'C']
    cm = confusion_matrix(all_truths, all_preds, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Cascade SSP Confusion Matrix")
    plt.tight_layout()
    plt.savefig("cascade_ssp_confusion.png")

if __name__ == "__main__":
    main()