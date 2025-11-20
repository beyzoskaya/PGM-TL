import os
import torch
import json
import numpy as np

from protbert_hf import SharedProtBert
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from engine_hf_with_task_specific_encoder import MultiTaskEngine, set_seed, ensure_dir

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SANITY_CHECK = True      
EPOCHS = 5
BATCH_SIZE = 16
MAX_LENGTH = 512
SEED = 42
SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/uncertainty_weighting_v2"

set_seed(SEED)
ensure_dir(SAVE_DIR)

print("\n[DATA] Loading datasets...")
# Load each dataset once, then split into train/valid/test
thermo_full = Thermostability(verbose=1)
thermo_train, thermo_valid, thermo_test = thermo_full.split()

ssp_full = SecondaryStructure(verbose=1)
ssp_train, ssp_valid, ssp_test = ssp_full.split()

clf_full = CloningCLF(verbose=1)
clf_train, clf_valid, clf_test = clf_full.split()

print(f"\n[DATA] Thermostability: train={len(thermo_train)}, valid={len(thermo_valid)}, test={len(thermo_test)}")
print(f"[DATA] SecondaryStructure: train={len(ssp_train)}, valid={len(ssp_valid)}, test={len(ssp_test)}")
print(f"[DATA] CloningCLF: train={len(clf_train)}, valid={len(clf_valid)}, test={len(clf_test)}")

train_sets = [thermo_train, ssp_train, clf_train]
valid_sets = [thermo_valid, ssp_valid, clf_valid]
test_sets = [thermo_test, ssp_test, clf_test]

task_configs = [
    {'type': 'regression', 'num_labels': 1, 'name': 'Thermostability'},
    {'type': 'per_residue_classification', 'num_labels': 8, 'name': 'SecondaryStructure'},
    {'type': 'sequence_classification', 'num_labels': 2, 'name': 'CloningCLF'}
]

print("\n[MODEL] Initializing Shared ProtBERT with LoRA...")
backbone = SharedProtBert(
    model_name="Rostlab/prot_bert_bfd",
    readout="mean",
    lora=True,
    lora_rank=32,
    lora_alpha=32,
    lora_dropout=0.1,
    freeze_backbone=True,
    unfrozen_layers=4,
    verbose=True
)

print("\n[ENGINE] Creating MultiTaskEngine...")
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
    verbose=True,
    grad_clip=1.0
)

# Optimizer includes LoRA parameters + task heads + uncertainty weights
optimizer = torch.optim.Adam(
    list(backbone.parameters()) + 
    list(engine.task_heads.parameters()) + 
    [engine.log_vars],
    lr=1e-3,
    weight_decay=1e-5
)

def safe_history(history_dict):
    """Convert tensors to serializable format"""
    def convert(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        elif isinstance(obj, (np.ndarray, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, list):
            return [convert(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        else:
            return obj
    return convert(history_dict)

if SANITY_CHECK:
    print("\n" + "="*60)
    print("[SANITY CHECK] Running 1 batch per task")
    print("="*60)

    avg_loss = engine.train_one_epoch(optimizer, max_batches_per_loader=1)
    print(f"[SANITY] Training loss: {avg_loss:.4f}")
    
    val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=0)
    print(f"[SANITY] Validation metrics: {val_metrics}")

    # Save sanity check results
    log_vars_safe = [float(lv) for lv in engine.log_vars]
    grad_norms_safe = engine.history["gradient_norms"][-1] if engine.history["gradient_norms"] else None

    engine.history["train_loss"].append(float(avg_loss))
    engine.history["val_metrics"].append(val_metrics)
    engine.history["log_vars"].append(log_vars_safe)

    sanity_history_path = os.path.join(SAVE_DIR, "sanity_check_history.json")
    with open(sanity_history_path, 'w') as f:
        json.dump(safe_history(engine.history), f, indent=2)
    print(f"✓ Sanity check history saved to {sanity_history_path}")

    sanity_model_path = os.path.join(SAVE_DIR, "sanity_check_model.pt")
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'task_heads_state_dict': engine.task_heads.state_dict(),
        'log_vars': engine.log_vars.detach().cpu(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 0
    }, sanity_model_path)
    print(f"✓ Sanity check model saved to {sanity_model_path}")
    
    print("[SANITY] Attempting to reload model...")
    checkpoint = torch.load(sanity_model_path, map_location=DEVICE)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    engine.task_heads.load_state_dict(checkpoint['task_heads_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("✓ Model reloaded successfully!")
    
    print("\n[SANITY] Sanity check passed! Exiting...")
    exit(0)

# Full training loop
print("\n" + "="*60)
print(f"STARTING FULL TRAINING: {EPOCHS} epochs")
print("="*60)

for epoch in range(EPOCHS):
    print(f"\n[EPOCH {epoch+1}/{EPOCHS}]")
    print("-" * 60)

    # Train
    avg_loss = engine.train_one_epoch(optimizer)
    print(f"✓ Training loss: {avg_loss:.4f}")

    # Validate
    val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=epoch+1)

    # Save history
    log_vars_safe = [float(lv) for lv in engine.log_vars]
    grad_norms_safe = engine.history["gradient_norms"][-1] if engine.history["gradient_norms"] else None

    engine.history["train_loss"].append(float(avg_loss))
    engine.history["val_metrics"].append(val_metrics)
    engine.history["log_vars"].append(log_vars_safe)

    # Save checkpoint
    ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch+1}.pt")
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'task_heads_state_dict': engine.task_heads.state_dict(),
        'log_vars': engine.log_vars.detach().cpu(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch+1
    }, ckpt_path)
    print(f"✓ Checkpoint saved: {ckpt_path}")

    # Save history
    history_path = os.path.join(SAVE_DIR, "history.json")
    with open(history_path, 'w') as f:
        json.dump(safe_history(engine.history), f, indent=2)

# Final test evaluation
print("\n" + "="*60)
print("[TEST] Final evaluation on test set")
print("="*60)
test_metrics = engine.evaluate(engine.test_loaders, split_name="Test")
engine.history["test_metrics"] = test_metrics

# Save final results
final_path = os.path.join(SAVE_DIR, "final_results.json")
with open(final_path, 'w') as f:
    json.dump(safe_history(engine.history), f, indent=2)
print(f"✓ Final results saved: {final_path}")

engine.print_best_metrics()

print("\n✓ Training complete!")