import torch
from protbert_hf import SharedProtBert       
from flip_hf import Thermostability, SecondaryStructure, CloningCLF  
from engine_hf_with_task_specific_encoder import MultiTaskEngine
from torch.utils.data import Subset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------
# Load datasets (full)
# ----------------------------
thermo_ds = Thermostability(split='train')
ssp_ds = SecondaryStructure(split='train')
clf_ds = CloningCLF(split='train')

# ----------------------------
# Take small portions for fast testing
# ----------------------------
thermo_ds_train = Subset(thermo_ds, range(100))
ssp_ds_train = Subset(ssp_ds, range(100))
clf_ds_train = Subset(clf_ds, range(100))

thermo_ds_valid = Subset(thermo_ds, range(20))
ssp_ds_valid = Subset(ssp_ds, range(20))
clf_ds_valid = Subset(clf_ds, range(20))

thermo_ds_test = Subset(thermo_ds, range(20))
ssp_ds_test = Subset(ssp_ds, range(20))
clf_ds_test = Subset(clf_ds, range(20))

# ----------------------------
# Task configs
# ----------------------------
task_configs = [
    {'type': 'regression', 'num_labels': 1},                   # Thermostability
    {'type': 'per_residue_classification', 'num_labels': 8},   # SSP Q8
    {'type': 'classification', 'num_labels': 2}                # CloningCLF
]

# ----------------------------
# Initialize backbone
# ----------------------------
backbone = SharedProtBert().to(device)

# ----------------------------
# Initialize engine
# ----------------------------
engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=task_configs,
    train_sets=[thermo_ds_train, ssp_ds_train, clf_ds_train],
    valid_sets=[thermo_ds_valid, ssp_ds_valid, clf_ds_valid],
    test_sets=[thermo_ds_test, ssp_ds_test, clf_ds_test],
    batch_size=2,
    device=device
)

# ----------------------------
# Test first batch for each task
# ----------------------------
for idx, name in enumerate(['Thermo', 'SSP', 'CloningCLF']):
    print(f"\n--- Dataset {idx} ({name}) first batch ---")
    batch = next(iter(engine.train_loaders[idx]))
    logits, targets = engine.forward(batch, dataset_idx=idx)
    print(f"=== Dataset {idx} ({name}) test complete ===\n")
