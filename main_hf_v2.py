import os
import torch
import json
import numpy as np

from protbert_hf import SharedProtBert
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from engine_hf_with_task_specific_encoder import MultiTaskEngine, set_seed, ensure_dir

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SANITY_CHECK = True
EPOCHS = 10
BATCH_SIZE = 16
MAX_LENGTH = 512
SEED = 42
SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/simplified_v3"

# HYPERPARAMETERS - OPTIMIZED FOR LEARNING
LR = 5e-4              # Reduced from 1e-3
WEIGHT_DECAY = 1e-4    # Added for regularization
GRAD_CLIP = 1.0
LORA_RANK = 16         # Reduced from 32
LORA_ALPHA = 16        # Match rank
LORA_DROPOUT = 0.05    # Reduced from 0.1
UNFROZEN_LAYERS = 2    # Reduced from 4
TASK_WEIGHTS = [1.0, 1.5, 1.0]  # Help Task 1 (harder) learn more

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
print(f"  LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"  Unfrozen layers={UNFROZEN_LAYERS}")
print(f"  Task weights={TASK_WEIGHTS}")

backbone = SharedProtBert(
    model_name="Rostlab/prot_bert_bfd",
    readout="mean",
    lora=True,
    lora_rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    freeze_backbone=True,
    unfrozen_layers=UNFROZEN_LAYERS,
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
    grad_clip=GRAD_CLIP,
    task_weights=TASK_WEIGHTS
)

# Optimizer: only include backbone params + task head params (no log_vars now)
optimizer = torch.optim.Adam(
    list(backbone.parameters()) + 
    list(engine.task_heads.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

print(f"\n[OPTIMIZER] lr={LR}, weight_decay={WEIGHT_DECAY}, grad_clip={GRAD_CLIP}")

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
    print(f"✓ Training loss: {avg_loss:.4f}")
    
    val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=0)

    # Save sanity check results
    engine.history["train_loss"].append(float(avg_loss))
    engine.history["val_metrics"].append(val_metrics)

    sanity_history_path = os.path.join(SAVE_DIR, "sanity_check_history.json")
    with open(sanity_history_path, 'w') as f:
        json.dump(safe_history(engine.history), f, indent=2)
    print(f"✓ Sanity check history saved")

    sanity_model_path = os.path.join(SAVE_DIR, "sanity_check_model.pt")
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'task_heads_state_dict': engine.task_heads.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 0
    }, sanity_model_path)
    print(f"✓ Sanity check model saved")
    
    checkpoint = torch.load(sanity_model_path, map_location=DEVICE)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    engine.task_heads.load_state_dict(checkpoint['task_heads_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("✓ Model reload successful!")
    
    print("\n✓ Sanity check PASSED! Ready for full training.\n")
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

    # Validate
    val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=epoch+1)

    # Save history
    engine.history["train_loss"].append(float(avg_loss))
    engine.history["val_metrics"].append(val_metrics)

    # Save checkpoint
    ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch+1}.pt")
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'task_heads_state_dict': engine.task_heads.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch+1
    }, ckpt_path)
    print(f"✓ Checkpoint saved")

    # Save history
    history_path = os.path.join(SAVE_DIR, "history.json")
    with open(history_path, 'w') as f:
        json.dump(safe_history(engine.history), f, indent=2)

    # Print summary
    if epoch > 0:
        prev_loss = engine.history["train_loss"][-2]
        curr_loss = engine.history["train_loss"][-1]
        improvement = (prev_loss - curr_loss) / prev_loss * 100
        print(f"Loss improvement: {improvement:.2f}%")

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
print(f"✓ Final results saved")

engine.print_best_metrics()

print("\n✓ Training complete!")
print(f"✓ Results saved to: {SAVE_DIR}")