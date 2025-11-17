import torch
from torch import nn, optim

from protbert_hf import SharedProtBert
from flip_hf import Thermostability, SecondaryStructure, CloningCLF 
from engine_hf_with_task_specific_encoder import MultiTaskEngine

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======================
# 1. Load datasets
# ======================
thermo_ds = Thermostability(split='train')  # can select subset if needed
ssp_ds = SecondaryStructure(split='train')
clf_ds = CloningCLF(split='train')

thermo_ds_train = thermo_ds.select(range(100))
ssp_ds_train = ssp_ds.select(range(100))
clf_ds_train = clf_ds.select(range(100))

train_sets = [thermo_ds_train, ssp_ds_train, clf_ds_train]
valid_sets = train_sets  # just reuse for test
test_sets = train_sets   # just reuse for test

# ======================
# 2. Task configs
# ======================
task_configs = [
    {'type': 'regression', 'num_labels': 1},             # Thermostability
    {'type': 'per_residue_classification', 'num_labels': 8},  # SSP (Q8)
    {'type': 'classification', 'num_labels': 2}          # CloningCLF
]

# ======================
# 3. Initialize backbone and engine
# ======================
backbone = SharedProtBert()
backbone.to(device)

engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=task_configs,
    train_sets=train_sets,
    valid_sets=valid_sets,
    test_sets=test_sets,
    batch_size=2,
    device=device
)

# ======================
# 4. Optimizer
# ======================
optimizer = optim.Adam(list(engine.backbone.parameters()) + list(engine.task_heads.parameters()),
                       lr=1e-4)

# ======================
# 5. Train one epoch (fast test)
# ======================
# max_batches_per_task limits the run to 2 batches per task
engine.train_one_epoch(optimizer, max_batches_per_task=2)

print("\n✅ MultiTaskEngine train_one_epoch test completed successfully!")
