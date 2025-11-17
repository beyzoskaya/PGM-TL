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
# Subset for fast testing
# ----------------------------
thermo_ds_train = Subset(thermo_ds, range(100))
ssp_ds_train = Subset(ssp_ds, range(100))
clf_ds_train = Subset(clf_ds, range(100))

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
    batch_size=2,
    device=device
)

# ----------------------------
# Optimizer
# ----------------------------
optimizer = optim.Adam(list(backbone.parameters()) + list(engine.task_heads.parameters()), lr=1e-4)

# ----------------------------
# Run one epoch for testing
# ----------------------------
engine.train_one_epoch(optimizer, max_batches_per_task=2)  # limit to 2 batches per task for speed
