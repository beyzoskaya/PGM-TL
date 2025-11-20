import os
import torch
import json
import numpy as np
from itertools import cycle

from protbert_hf import SharedProtBert, build_regression_head, build_token_classification_head, build_sequence_classification_head
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from engine_hf_with_task_specific_encoder import MultiTaskEngine, set_seed, ensure_dir

# ====================== CONFIG ======================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SANITY_CHECK = False
EPOCHS = 5
BATCH_SIZE = 16
MAX_LENGTH = 512
SEED = 42
DEBUG_INTERVAL = 100
SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/dynamic_weighting"

set_seed(SEED)

print("\n[DATA] Loading datasets...")
thermo_train = Thermostability(split='train')
thermo_valid = Thermostability(split='valid')
thermo_test  = Thermostability(split='test')

ssp_train = SecondaryStructure(split='train')
ssp_valid = SecondaryStructure(split='valid')
ssp_test  = SecondaryStructure(split='test')

clf_train = CloningCLF(split='train')
clf_valid = CloningCLF(split='valid')
clf_test  = CloningCLF(split='test')

train_sets = [thermo_train, ssp_train, clf_train]
valid_sets = [thermo_valid, ssp_valid, clf_valid]
test_sets  = [thermo_test,  ssp_test,  clf_test]

task_configs = [
    {'type':'regression', 'num_labels':1, 'name':'Thermostability'},
    {'type':'per_residue_classification', 'num_labels':8, 'name':'SecondaryStructure'},
    {'type':'sequence_classification', 'num_labels':2, 'name':'CloningCLF'}
]

print("\n[MODEL] Initializing Shared ProtBERT with LoRA (PEFT)...")
backbone = SharedProtBert(
    model_name="Rostlab/prot_bert_bfd",
    readout="mean",
    lora=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1,
    freeze_backbone=True,
    verbose=True
)

engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=task_configs,
    train_sets=train_sets,
    valid_sets=valid_sets,
    test_sets=test_sets,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    save_dir=SAVE_DIR,
    max_length=MAX_LENGTH,
    verbose=True
)

if SANITY_CHECK:
    print("\n[SANITY CHECK] Forward + backward pass for one batch per task")
    engine.train_one_epoch(
        optimizer=torch.optim.Adam(list(backbone.parameters()) + list(engine.task_heads.parameters()), lr=1e-3),
        max_batches_per_loader=1
    )
    engine.evaluate(engine.valid_loaders)
    print("\n✓ Sanity check completed successfully\n")
    exit(0)

optimizer = torch.optim.Adam(
    list(backbone.parameters()) + list(engine.task_heads.parameters()),
    lr=1e-3
)

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    
    # 1️⃣ Train one epoch
    avg_loss = engine.train_one_epoch(optimizer)
    print(f"[Epoch {epoch+1}] Average training loss: {avg_loss:.4f}")

    # 2️⃣ Evaluate on validation
    val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=epoch)
    
    # 3️⃣ Update dynamic weights EMA
    dyn_weights = engine.update_dynamic_weights_epoch()

    # 4️⃣ Append metrics/history
    engine.history["train_loss"].append(avg_loss)
    engine.history["val_metrics"].append(val_metrics)
    engine.history["dynamic_weights"].append(dyn_weights)
    engine.history["gradient_norms"].append(engine.gradient_norms_log[-1])

    # 5️⃣ Save intermediate history
    history_path = os.path.join(SAVE_DIR, f"history_epoch{epoch+1}.json")
    with open(history_path, 'w') as f:
        json.dump(engine.history, f, indent=2)
    print(f"[INFO] Saved training history to {history_path}")

print("\n[TEST] Evaluating final model on test sets...")
test_metrics = engine.evaluate(engine.test_loaders, split_name="Test")
engine.history["test_metrics"] = test_metrics

final_metrics_path = os.path.join(SAVE_DIR, "final_metrics.json")
with open(final_metrics_path, 'w') as f:
    json.dump(test_metrics, f, indent=2)
print(f"[INFO] Saved test metrics to {final_metrics_path}")

engine.print_best_metrics()
