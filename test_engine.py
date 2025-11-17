import torch
from flip_hf import ThermostabilityDataset, SecondaryStructureDataset, CloningCLFDataset
from protbert_hf import SharedProtBert
from engine_hf_with_task_specific_encoder import MultiTaskEngine

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 2  # small batch for fast testing

# --- 1. Load datasets ---
thermo_dataset = ThermostabilityDataset(split='train')
ssp_dataset = SecondaryStructureDataset(split='train')
cloning_dataset = CloningCLFDataset(split='train')

train_sets = [thermo_dataset, ssp_dataset, cloning_dataset]
valid_sets = [thermo_dataset, ssp_dataset, cloning_dataset]  # can reuse for quick test
test_sets  = [thermo_dataset, ssp_dataset, cloning_dataset]

# --- 2. Create backbone ---
backbone = SharedProtBert()
backbone = backbone.to(device)

# --- 3. Task configs (for reference, not used yet) ---
task_configs = [
    {'type': 'regression', 'num_labels': 1},      # Thermostability
    {'type': 'classification', 'num_labels': 8},  # SecondaryStructure Q8
    {'type': 'classification', 'num_labels': 2}   # CloningCLF
]

# --- 4. Initialize engine ---
engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=task_configs,
    train_sets=train_sets,
    valid_sets=valid_sets,
    test_sets=test_sets,
    batch_size=batch_size,
    device=device
)

# --- 5. Iterate over first batch of each dataset and do forward pass ---
for idx, loader in enumerate(engine.train_loaders):
    print(f"\n--- Dataset {idx} first batch ---")
    batch = next(iter(loader))
    embeddings, targets = engine.forward(batch, dataset_idx=idx)
