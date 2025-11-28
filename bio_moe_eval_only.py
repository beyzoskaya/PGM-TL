import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_bio_moe import BioMoE_Engine, BioPropertyFeaturizer

EPOCH_TO_EVAL = 4
SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/bio_moe_v1"
PLOT_DIR = os.path.join(SAVE_DIR, "final_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LORA_RANK = 16
UNFROZEN_LAYERS = 0

def normalize_regression_targets(train_ds):
    full_dataset = train_ds.dataset; train_indices = train_ds.indices
    all_raw_targets = full_dataset.targets['target']
    train_values = [all_raw_targets[i] for i in train_indices if all_raw_targets[i] is not None]
    mean = np.mean(train_values); std = np.std(train_values)
    
    new_targets = []
    for t in all_raw_targets:
        if t is None: new_targets.append(None)
        else: new_targets.append((t - mean) / std)
    full_dataset.targets['target'] = new_targets
    return mean, std

def run_inference(engine, loader_list):
    """
    Runs inference once and collects EVERYTHING:
    - Predictions, Truths
    - Router Weights
    - Biological Features (GRAVY, Charge, etc.) for analysis
    """
    engine.eval()

    data = {
        'Thermo': {'true': [], 'pred': [], 'weights': [], 'bio_feats': []},
        'SSP':    {'true': [], 'pred': [], 'weights': []}, # SSP is token-level, skipping bio-feats per token for now
        'Clone':  {'true': [], 'pred': [], 'prob': [], 'weights': [], 'bio_feats': []}
    }

    print("🚀 Running Inference on Test Set...")
    
    with torch.no_grad():
        for i, loader in enumerate(loader_list):
            task_key = ['Thermo', 'SSP', 'Clone'][i]
            
            for batch in tqdm(loader, desc=f"Scanning {task_key}"):
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                raw_seqs = batch['raw_sequences']
                
                # Get Predictions AND Router Weights
                logits, router_weights = engine.forward(input_ids, mask, raw_seqs, task_idx=i)
                
                # Calculate Bio Features manually for correlation plots later
                bio_feats = BioPropertyFeaturizer.get_features(raw_seqs, DEVICE).cpu().numpy()
                
                # Move weights to CPU
                rw = router_weights.cpu().numpy()
                
                if i == 0: # Thermo
                    data['Thermo']['true'].extend(targets.view(-1).cpu().numpy())
                    data['Thermo']['pred'].extend(logits.view(-1).cpu().numpy())
                    data['Thermo']['weights'].extend(rw)
                    data['Thermo']['bio_feats'].extend(bio_feats)
                    
                elif i == 1: # SSP
                    p = logits.argmax(dim=-1).cpu().numpy()
                    t = targets.cpu().numpy()
                    # Flatten for metrics
                    for b in range(t.shape[0]):
                        valid = t[b] != -100
                        data['SSP']['true'].extend(t[b][valid])
                        data['SSP']['pred'].extend(p[b][valid])
                    # Store average router weight for this batch
                    data['SSP']['weights'].extend(rw)

                elif i == 2: # Cloning
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    data['Clone']['true'].extend(targets.cpu().numpy())
                    data['Clone']['pred'].extend(logits.argmax(dim=1).cpu().numpy())
                    data['Clone']['prob'].extend(probs.cpu().numpy())
                    data['Clone']['weights'].extend(rw)
                    data['Clone']['bio_feats'].extend(bio_feats)

    return data

def plot_performance_dashboard(data):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Thermostability Scatter
    ax = axes[0]
    y_true = data['Thermo']['true']
    y_pred = data['Thermo']['pred']
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.3, color="#7B2CBF")
    # Ideal line
    low, high = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    ax.plot([low, high], [low, high], 'k--', lw=2, label="Ideal")
    ax.set_title("Thermostability (Regression)")
    ax.set_xlabel("True Stability (Z-Score)")
    ax.set_ylabel("Predicted Stability")
    
    # 2. SSP Confusion Matrix (Simplified)
    ax = axes[1]
    cm = confusion_matrix(data['SSP']['true'], data['SSP']['pred'], normalize='true')
    # Plot only first 3 classes (Helix, Sheet, Coil usually) to keep it readable, or full
    sns.heatmap(cm, ax=ax, cmap="Blues", annot=False)
    ax.set_title("Secondary Structure Confusion")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # 3. Cloning Histogram
    ax = axes[2]
    df_c = pd.DataFrame({"Prob": data['Clone']['prob'], "Truth": data['Clone']['true']})
    sns.histplot(data=df_c, x="Prob", hue="Truth", bins=20, ax=ax, palette={0: "red", 1: "green"}, kde=True)
    ax.set_title("Solubility Confidence Distribution")
    ax.set_xlabel("Predicted Probability")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "1_Performance_Dashboard.png"), dpi=300)
    print("Saved Plot 1")

def plot_router_brain(data):
    """Creates the Heatmap of which Task uses which Expert"""
    tasks = ['Thermo', 'SSP', 'Clone']
    avg_weights = []
    
    for t in tasks:
        # Stack all router weights for this task -> Mean
        w = np.array(data[t]['weights'])
        avg_weights.append(np.mean(w, axis=0))
        
    matrix = np.stack(avg_weights)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt=".2f",
                xticklabels=["Expert: Thermo", "Expert: Struct", "Expert: Clone"],
                yticklabels=tasks)
    plt.title("Bio-Router Attention Map\n(Rows=Tasks, Cols=Experts)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "2_Router_Brain.png"), dpi=300)
    print("Saved Plot 2")

def plot_biological_logic(data):
    """
    Advanced: Does the Router listen to Hydrophobicity?
    We plot: Hydrophobicity (X) vs Weight Assigned to Cloning Expert (Y)
    Hypothesis: Higher Hydrophobicity -> More reliance on Cloning Expert (to detect aggregation)
    """
    # Extract Cloning Task Data
    bio_feats = np.array(data['Clone']['bio_feats']) # [GRAVY, Charge, Arom]
    router_w = np.array(data['Clone']['weights'])    # [Exp_Thermo, Exp_Struct, Exp_Clone]
    
    gravy = bio_feats[:, 0]
    weight_clone_expert = router_w[:, 2] # Index 2 is Cloning Expert
    
    plt.figure(figsize=(10, 6))
    sns.regplot(x=gravy, y=weight_clone_expert, scatter_kws={'alpha':0.3, 'color':'teal'}, line_kws={'color':'red'})
    plt.title("Biological Logic Check: Hydrophobicity vs. Solubility Expert Usage")
    plt.xlabel("Sequence Hydrophobicity (GRAVY)")
    plt.ylabel("Router Weight assigned to Solubility Expert")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "3_Bio_Logic_Check.png"), dpi=300)
    print("Saved Plot 3")

def calculate_metrics(data):
    print("\n" + "="*30)
    print("FINAL TEST METRICS")
    print("="*30)
    
    # Thermo
    mse = mean_squared_error(data['Thermo']['true'], data['Thermo']['pred'])
    r, _ = pearsonr(data['Thermo']['true'], data['Thermo']['pred'])
    print(f"Thermostability:\n  MSE: {mse:.4f}\n  Pearson r: {r:.4f}")
    
    # SSP
    f1 = f1_score(data['SSP']['true'], data['SSP']['pred'], average='macro')
    acc = accuracy_score(data['SSP']['true'], data['SSP']['pred'])
    print(f"SecStructure:\n  Acc: {acc:.4f}\n  Macro F1: {f1:.4f}")
    
    # Cloning
    mcc = matthews_corrcoef(data['Clone']['true'], data['Clone']['pred'])
    acc = accuracy_score(data['Clone']['true'], data['Clone']['pred'])
    print(f"Cloning:\n  Acc: {acc:.4f}\n  MCC: {mcc:.4f}")
    print("="*30)

# --- MAIN ---
def main():
    # 1. Load Data
    print("Loading Data splits...")
    ds_t = Thermostability(verbose=0); t_train, _, t_test = ds_t.split()
    normalize_regression_targets(t_train)
    ds_s = SecondaryStructure(verbose=0); _, _, s_test = ds_s.split()
    ds_c = CloningCLF(verbose=0); _, _, c_test = ds_c.split()
    
    test_sets = [t_test, s_test, c_test]
    
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    # 2. Load Model
    print("Loading Model...")
    backbone = SharedProtBert(lora_rank=LORA_RANK, unfrozen_layers=UNFROZEN_LAYERS)
    engine = BioMoE_Engine(backbone, task_configs, [], [], [], device=DEVICE)
    
    weights_path = os.path.join(SAVE_DIR, f"model_epoch_{EPOCH_TO_EVAL}.pt")
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights not found at {weights_path}")
        return
        
    engine.load_state_dict(torch.load(weights_path, map_location=DEVICE), strict=False)
    
    # 3. Create Loaders
    test_loaders = [torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, 
                    collate_fn=lambda b: engine.train_loaders[0].collate_fn(b)) for ds in test_sets]
    # Note: We reuse the collate_fn from the engine class logic or recreate it. 
    # To be safe, let's just use the function directly from engine_hf_bio_moe file if imported
    from engine_hf_bio_moe import multitask_collate_fn
    test_loaders = [torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, 
                    collate_fn=lambda b: multitask_collate_fn(b, backbone.tokenizer)) for ds in test_sets]

    # 4. Execute
    raw_data = run_inference(engine, test_loaders)
    calculate_metrics(raw_data)
    
    print("\nGenerating Visualizations...")
    plot_performance_dashboard(raw_data)
    plot_router_brain(raw_data)
    plot_biological_logic(raw_data)
    
    print(f"\n✅ Evaluation Done! Check {PLOT_DIR}")

if __name__ == "__main__":
    main()