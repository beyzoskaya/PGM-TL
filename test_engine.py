import torch
from torch import optim
from torch.utils.data import Subset

from protbert_hf import SharedProtBert
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from engine_hf_with_task_specific_encoder import MultiTaskEngine

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------
# Load datasets
# ----------------------------
thermo_ds = Thermostability(split='train')
ssp_ds = SecondaryStructure(split='train')
clf_ds = CloningCLF(split='train')

# ----------------------------
# Subset for fast testing (or increase for larger batch experiments)
# ----------------------------
subset_size = 500 # adjust for debugging
thermo_ds_train = Subset(thermo_ds, range(subset_size))
ssp_ds_train = Subset(ssp_ds, range(subset_size))
clf_ds_train = Subset(clf_ds, range(subset_size))

# ----------------------------
# Task configs
# ----------------------------
task_configs = [
    {'type': 'regression', 'num_labels': 1},                   # Thermostability
    {'type': 'per_residue_classification', 'num_labels': 8},   # SSP Q8
    {'type': 'classification', 'num_labels': 2}                # CloningCLF
]

# ----------------------------
# Backbone and engine
# ----------------------------
backbone = SharedProtBert().to(device)

engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=task_configs,
    train_sets=[thermo_ds_train, ssp_ds_train, clf_ds_train],
    valid_sets=[thermo_ds_train, ssp_ds_train, clf_ds_train],  # reuse subset
    test_sets=[thermo_ds_train, ssp_ds_train, clf_ds_train],
    batch_size=16,  # try larger batch size
    device=device
)

# ----------------------------
# Optimizer
# ----------------------------
optimizer = optim.Adam(
    list(backbone.parameters()) + list(engine.task_heads.parameters()),
    lr=1e-4
)

# ----------------------------
# Debugging dataset
# ----------------------------
print("=== Dataset Samples ===")
for i, ds in enumerate([thermo_ds_train, ssp_ds_train, clf_ds_train]):
    sample = ds[0]
    print(f"Task {i} sample keys:", sample.keys())
    print(f"Sequence length: {len(sample['sequence'])}")
    print(f"Targets: {sample['targets']}")

# ----------------------------
# Test forward pass with one batch per task
# ----------------------------
print("\n=== Forward Pass Debug ===")
for task_idx, loader in enumerate(engine.train_loaders):
    batch = next(iter(loader))
    logits, targets = engine.forward(batch, dataset_idx=task_idx)
    target_tensor = list(targets.values())[0]

    print(f"\nTask {task_idx} ({task_configs[task_idx]['type']}):")
    print("Input batch size:", batch[0]['input_ids'].shape if 'input_ids' in batch[0] else 'N/A')
    print("Logits shape:", logits.shape)
    print("Target shape:", target_tensor.shape)

    # Compute loss for inspection
    task_cfg = task_configs[task_idx]
    if task_cfg['type'] == 'regression':
        loss = torch.nn.MSELoss()(logits, target_tensor)
    elif task_cfg['type'] == 'classification':
        loss = torch.nn.CrossEntropyLoss()(logits, target_tensor.squeeze(-1).long())
    elif task_cfg['type'] == 'per_residue_classification':
        b, s, c = logits.shape
        loss = torch.nn.CrossEntropyLoss()(logits.view(b*s, c), target_tensor.view(b*s))
    print("Batch loss:", loss.item())

# ----------------------------
# Optional: run one training epoch to see dynamic weighting in action
# ----------------------------
print("\n=== Training Epoch Debug ===")
engine.train_one_epoch(optimizer, max_batches_per_task=10)  # small number of batches for debug
