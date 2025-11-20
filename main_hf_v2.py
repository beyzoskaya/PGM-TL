import os
import torch
import json
import numpy as np

from protbert_hf import SharedProtBert
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from engine_hf_with_task_specific_encoder import MultiTaskEngine, set_seed, ensure_dir

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SANITY_CHECK = False  
EPOCHS = 50
BATCH_SIZE = 16
MAX_LENGTH = 512
SEED = 42
SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/cyclic_v1"

# ============================================================================
# HYPERPARAMETERS - OPTIMIZED FOR CYCLIC MULTI-TASK LEARNING
# ============================================================================
LR = 3e-4              # Reduced learning rate for stability
WEIGHT_DECAY = 1e-4    # L2 regularization
GRAD_CLIP = 1.0

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
UNFROZEN_LAYERS = 4

# CRITICAL: Task weights should be EQUAL (1.0) because:
# - Loss normalization handles magnitude differences
# - Per-residue losses divided by num_valid_tokens
# - Cyclic sampling ensures balanced updates
TASK_WEIGHTS = [1.0, 1.0, 1.0]

# ============================================================================

set_seed(SEED)
ensure_dir(SAVE_DIR)

print("\n" + "="*70)
print("MULTI-TASK PROTEIN LEARNING - CYCLIC BATCH STRATEGY")
print("="*70)

print("\n[DATA] Loading datasets...")
# Load each dataset once, then split into train/valid/test
thermo_full = Thermostability(verbose=1)
thermo_train, thermo_valid, thermo_test = thermo_full.split()

ssp_full = SecondaryStructure(verbose=1)
ssp_train, ssp_valid, ssp_test = ssp_full.split()

clf_full = CloningCLF(verbose=1)
clf_train, clf_valid, clf_test = clf_full.split()

print(f"\n[DATA] Dataset sizes:")
print(f"  Thermostability:    train={len(thermo_train):4d}, valid={len(thermo_valid):4d}, test={len(thermo_test):4d}")
print(f"  SecondaryStructure: train={len(ssp_train):4d}, valid={len(ssp_valid):4d}, test={len(ssp_test):4d}")
print(f"  CloningCLF:         train={len(clf_train):4d}, valid={len(clf_valid):4d}, test={len(clf_test):4d}")

train_sets = [thermo_train, ssp_train, clf_train]
valid_sets = [thermo_valid, ssp_valid, clf_valid]
test_sets = [thermo_test, ssp_test, clf_test]

task_configs = [
    {'type': 'regression', 'num_labels': 1, 'name': 'Thermostability'},
    {'type': 'per_residue_classification', 'num_labels': 8, 'name': 'SecondaryStructure'},
    {'type': 'sequence_classification', 'num_labels': 2, 'name': 'CloningCLF'}
]

print("\n[MODEL] Initializing Shared ProtBERT with LoRA...")
print(f"  Model: Rostlab/prot_bert_bfd")
print(f"  LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"  Unfrozen layers={UNFROZEN_LAYERS}")

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

print("\n[ENGINE] Calculating epoch parameters...")

# Pre-calculate batch information
train_loader_sizes = []
for ds in train_sets:
    # Simulate DataLoader length calculation
    dataset_size = len(ds)
    num_batches = (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE
    train_loader_sizes.append(num_batches)
    
max_batches = max(train_loader_sizes)
total_updates_per_epoch = max_batches * len(task_configs)

print(f"  Dataset → Batch counts: {train_loader_sizes}")
print(f"  Max batches: {max_batches}")
print(f"  Total updates/epoch: {total_updates_per_epoch:,}")
print(f"  Epoch strategy: Cyclic round-robin (shorter loaders cycle)")

print("\n[ENGINE] Creating MultiTaskEngine with cyclic batch strategy...")
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

# Create optimizer: backbone + task heads
optimizer = torch.optim.Adam(
    list(backbone.parameters()) + 
    list(engine.task_heads.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

print(f"\n[OPTIMIZER] lr={LR}, weight_decay={WEIGHT_DECAY}, grad_clip={GRAD_CLIP}")

def safe_history(history_dict):
    """Convert tensors to serializable format for JSON"""
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

# ============================================================================
# SANITY CHECK: Verify everything works with 1 batch per task
# ============================================================================
if SANITY_CHECK:
    print("\n" + "="*70)
    print("[SANITY CHECK] Running 1 epoch with cyclic batches")
    print("="*70)

    avg_loss = engine.train_one_epoch(optimizer)
    print(f"✓ Training loss: {avg_loss:.4f}")
    
    val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=0)
    print(f"✓ Validation completed")

    # Save sanity check results
    engine.history["train_loss"].append(float(avg_loss))
    engine.history["val_metrics"].append(val_metrics)

    sanity_history_path = os.path.join(SAVE_DIR, "sanity_check_history.json")
    with open(sanity_history_path, 'w') as f:
        json.dump(safe_history(engine.history), f, indent=2)
    print(f"✓ Sanity check history saved to {sanity_history_path}")

    sanity_model_path = os.path.join(SAVE_DIR, "sanity_check_model.pt")
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'task_heads_state_dict': engine.task_heads.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 0
    }, sanity_model_path)
    print(f"✓ Sanity check model saved")
    
    # Test model loading
    checkpoint = torch.load(sanity_model_path, map_location=DEVICE)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    engine.task_heads.load_state_dict(checkpoint['task_heads_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("✓ Model reload successful!")
    
    print("\n✓ SANITY CHECK PASSED! Ready for full training.\n")
    exit(0)

# ============================================================================
# FULL TRAINING LOOP
# ============================================================================
print("\n" + "="*70)
print(f"STARTING FULL TRAINING: {EPOCHS} epochs")
print("="*70)

best_overall_loss = float('inf')

for epoch in range(EPOCHS):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch+1}/{EPOCHS}")
    print(f"{'='*70}")

    # Train one epoch (cyclic batches)
    avg_loss = engine.train_one_epoch(optimizer)
    engine.history["train_loss"].append(float(avg_loss))

    # Validate on all tasks
    print(f"\n[VALIDATION]")
    val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=epoch+1)
    engine.history["val_metrics"].append(val_metrics)

    # Save checkpoint
    ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch+1}.pt")
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'task_heads_state_dict': engine.task_heads.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch+1,
        'best_scores': engine.best
    }, ckpt_path)
    print(f"✓ Checkpoint saved to {ckpt_path}")

    # Save history after each epoch
    history_path = os.path.join(SAVE_DIR, "history.json")
    with open(history_path, 'w') as f:
        json.dump(safe_history(engine.history), f, indent=2)

    # Print training summary
    if epoch > 0:
        prev_loss = engine.history["train_loss"][-2]
        curr_loss = engine.history["train_loss"][-1]
        improvement = ((prev_loss - curr_loss) / prev_loss) * 100 if prev_loss > 0 else 0
        print(f"\nTrain loss: {curr_loss:.4f} (improvement: {improvement:+.2f}%)")
    
    # Track best overall loss
    if avg_loss < best_overall_loss:
        best_overall_loss = avg_loss
        best_model_path = os.path.join(SAVE_DIR, "best_model.pt")
        torch.save({
            'backbone_state_dict': backbone.state_dict(),
            'task_heads_state_dict': engine.task_heads.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch+1,
            'best_loss': best_overall_loss,
            'best_scores': engine.best
        }, best_model_path)
        print(f"✓ New best model saved (loss: {best_overall_loss:.4f})")

# ============================================================================
# FINAL TEST EVALUATION
# ============================================================================
print("\n" + "="*70)
print("[TEST] Final evaluation on test set")
print("="*70)

test_metrics = engine.evaluate(engine.test_loaders, split_name="Test")
engine.history["test_metrics"] = test_metrics

# Save final results
final_path = os.path.join(SAVE_DIR, "final_results.json")
with open(final_path, 'w') as f:
    json.dump(safe_history(engine.history), f, indent=2)
print(f"✓ Final results saved to {final_path}")

# Print best metrics summary
engine.print_best_metrics()

# Summary stats
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Total epochs: {EPOCHS}")
print(f"Best training loss: {best_overall_loss:.4f}")
print(f"Save directory: {SAVE_DIR}")
print(f"✓ Training complete!\n")