# test_engine_step2.py
import torch
from protbert_hf import SharedProtBert   # your backbone
from flip_hf import Thermostability, SecondaryStructure, CloningCLF  
from engine_hf_with_task_specific_encoder import MultiTaskEngine

# ----------------------------
# Load small portion of each dataset for fast test
# ----------------------------
train_thermo = Thermostability(split='train', limit=5)  # take only 5 samples
train_ssp = SecondaryStructure(split='train', limit=5)
train_clf = CloningCLF(split='train', limit=5)

train_sets = [train_thermo, train_ssp, train_clf]
valid_sets = [train_thermo, train_ssp, train_clf]  # reuse same for test purposes
test_sets  = [train_thermo, train_ssp, train_clf]

# ----------------------------
# Task configs
# ----------------------------
task_configs = [
    {'type': 'regression', 'num_labels': 1},                  # Thermostability
    {'type': 'per_residue_classification', 'num_labels': 8},  # SSP Q8
    {'type': 'classification', 'num_labels': 2}               # CloningCLF
]

# ----------------------------
# Create backbone
# ----------------------------
backbone = SharedProtBert(lora=False)  # use regular ProtBert for testing
device = 'cuda' if torch.cuda.is_available() else 'cpu'
backbone.to(device)

# ----------------------------
# Initialize engine
# ----------------------------
engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=task_configs,
    train_sets=train_sets,
    valid_sets=valid_sets,
    test_sets=test_sets,
    batch_size=2,
    num_worker=0,
    device=device
)

# ----------------------------
# Test forward pass for each dataset
# ----------------------------
for i, name in enumerate(['Thermo', 'SSP', 'CloningCLF']):
    batch = next(iter(engine.train_loaders[i]))
    logits, targets = engine.forward(batch, dataset_idx=i)
    print(f"=== Dataset {i} ({name}) test complete ===\n")
