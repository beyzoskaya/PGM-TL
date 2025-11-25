import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_with_uncertanity_weighting import MultiTaskEngineUncertanityWeighting

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32 
LORA_RANK = 16
UNFROZEN_LAYERS = 0 

CHECKPOINT_PATH = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_uncertainty/model_epoch_2.pt"

def load_model_and_data():
    print("⏳ Loading Data and Model...")
    
    ds_thermo = Thermostability(verbose=0); _, _, thermo_test = ds_thermo.split()
    ds_ssp = SecondaryStructure(verbose=0); _, _, ssp_test = ds_ssp.split()
    ds_clf = CloningCLF(verbose=0); _, _, clf_test = ds_clf.split()

    backbone = SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=UNFROZEN_LAYERS)
    
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    
    engine = MultiTaskEngineUncertanityWeighting(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=[thermo_test], # Dummy, not training
        valid_sets=None,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    print(f"   Loading weights from: {CHECKPOINT_PATH}")
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    engine.load_state_dict(state_dict)
    engine.eval()
    engine.to(DEVICE)
    
    print("✅ Model Loaded Successfully.")
    return engine, thermo_test, ssp_test, clf_test

def plot_deep_mutational_scanning(engine, dataset):
    print("\n🔬 [Analysis 1] Running In-Silico Deep Mutational Scanning...")
    
    # 1. Pick a sequence (Length 100-300 is good for viz)
    target_idx = 0
    while True:
        seq = dataset[target_idx]['sequence']
        if 100 < len(seq) < 200: break
        target_idx += 1
        if target_idx > 100: break # Fallback
    
    print(f"   Target Protein Length: {len(seq)}")
    original_seq = list(seq)
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    
    # 2. Generate Mutants
    mutants = []
    heatmap_data = np.zeros((len(amino_acids), len(seq)))
    
    # Prepare batch of all mutants
    batch_seqs = []
    coordinates = [] # (aa_idx, pos_idx)
    
    for pos_i, original_aa in enumerate(original_seq):
        for aa_i, mutant_aa in enumerate(amino_acids):
            # Create mutant string
            mutant_seq_list = original_seq.copy()
            mutant_seq_list[pos_i] = mutant_aa
            mutant_str = "".join(mutant_seq_list)
            
            # Add spaces for ProtBERT
            batch_seqs.append(" ".join(list(mutant_str)))
            coordinates.append((aa_i, pos_i))

    # 3. Run Inference in Batches
    print(f"   Predicting {len(batch_seqs)} mutants...")
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(batch_seqs), BATCH_SIZE):
            batch = batch_seqs[i : i+BATCH_SIZE]
            inputs = engine.backbone.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(DEVICE)
            
            # Task 0 is Thermo
            emb = engine.backbone(inputs['input_ids'], inputs['attention_mask'], task_type='sequence')
            preds = engine.heads[0](emb).cpu().flatten().numpy()
            predictions.extend(preds)

    # 4. Fill Heatmap
    # We subtract the "Wild Type" (approximate) to see relative change (Delta Stability)
    # Or just plot absolute values. Let's plot Absolute Z-Score first.
    for pred, (aa_i, pos_i) in zip(predictions, coordinates):
        heatmap_data[aa_i, pos_i] = pred

    # 5. Plot
    plt.figure(figsize=(18, 6))
    sns.heatmap(heatmap_data, yticklabels=amino_acids, cmap="coolwarm", center=0)
    plt.title(f"In-Silico Mutational Scan (Thermostability Prediction)\nSequence Length: {len(seq)} residues")
    plt.xlabel("Residue Position")
    plt.ylabel("Mutation")
    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_uncertainty/deep_mutational_scanning.png", dpi=300)
    plt.show()
    print("   > Interpretation: Red = More Stable, Blue = Less Stable (Normalized Z-Score)")

# ==========================================
# ANALYSIS 2: CONFUSION MATRIX (SSP)
# ==========================================
def plot_ssp_confusion(engine, dataset):
    print("\n🧬 [Analysis 2] Generating Secondary Structure Confusion Matrix...")
    
    all_preds = []
    all_targets = []
    
    # Standard Collate for manual loader
    def collate(b): 
        from engine_hf_with_uncertanity_weighting import multitask_collate_fn
        return multitask_collate_fn(b, engine.backbone.tokenizer)

    loader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=collate)
    
    # Run on subset to save time (first 10 batches)
    print("   Running inference on SSP Test Set...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > 20: break # Analyze ~300 proteins
            
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            # Task 1 is SSP
            emb = engine.backbone(ids, mask, task_type='token')
            logits = engine.heads[1](emb)
            
            preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
            targs = targets.view(-1).cpu().numpy()
            
            # Filter Padding (-100)
            valid = targs != -100
            all_preds.extend(preds[valid])
            all_targets.extend(targs[valid])

    # Plot
    cm = confusion_matrix(all_targets, all_preds)
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
    plt.title("Secondary Structure Confusion Matrix (Q8)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.savefig("/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_uncertainty/ssp_confusion_matrix.png", dpi=300)
    plt.show()

# ==========================================
# ANALYSIS 3: STABILITY vs SOLUBILITY
# ==========================================
def plot_bio_correlation(engine, ds_thermo, ds_clf):
    print("\n⚗️ [Analysis 3] Stability vs. Solubility Correlation...")
    
    # We need a set of proteins where we predict BOTH.
    # Since test sets differ, let's just use the Thermo Test set
    # and predict Solubility for it (Hypothetical Solubility).
    
    thermo_preds = []
    solubility_probs = []
    
    # Create simple loader
    loader = torch.utils.data.DataLoader(ds_thermo, batch_size=32, 
                                         collate_fn=lambda b: {'input_ids': [x['sequence'] for x in b]})
    
    print("   Predicting joint properties...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > 30: break # ~1000 proteins
            
            # Manual Tokenization
            seqs = [" ".join(list(s)) for s in batch['input_ids']]
            inputs = engine.backbone.tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            
            # 1. Thermo (Sequence Task)
            emb_seq = engine.backbone(ids, mask, task_type='sequence')
            t_pred = engine.heads[0](emb_seq).flatten().cpu().numpy()
            thermo_preds.extend(t_pred)
            
            # 2. Cloning (Sequence Task) - Same embedding!
            # Task 2 is Cloning
            c_logits = engine.heads[2](emb_seq)
            c_probs = torch.softmax(c_logits, dim=1)[:, 1].cpu().numpy() # Prob of Class 1 (Soluble/Clonable)
            solubility_probs.extend(c_probs)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(thermo_preds, solubility_probs, alpha=0.5, s=10, c=solubility_probs, cmap='viridis')
    plt.title("Predicted Stability vs. Predicted Solubility")
    plt.xlabel("Predicted Thermostability (Z-Score)")
    plt.ylabel("Predicted Probability of Cloning Success")
    plt.grid(True, alpha=0.3)
    plt.colorbar(label="Solubility Prob")
    plt.savefig("/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1_lora16_uncertainty/stability_vs_solubility.png", dpi=300)
    plt.show()
    
    # Calc correlation
    corr = np.corrcoef(thermo_preds, solubility_probs)[0, 1]
    print(f"   > Pearson Correlation: {corr:.4f}")
    if corr > 0: print("   > Result: Positive Correlation! (More Stable = More Soluble)")
    else: print("   > Result: Negative/No Correlation.")

engine, thermo_ds, ssp_ds, clf_ds = load_model_and_data()

plot_deep_mutational_scanning(engine, thermo_ds)
plot_ssp_confusion(engine, ssp_ds)
plot_bio_correlation(engine, thermo_ds, clf_ds)