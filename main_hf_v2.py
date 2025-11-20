import os
import torch
import json
import numpy as np
from itertools import cycle

from protbert_hf import SharedProtBert, build_regression_head, build_token_classification_head, build_sequence_classification_head
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from engine_hf_with_task_specific_encoder import MultiTaskEngine, set_seed, ensure_dir

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SANITY_CHECK = True      
EPOCHS = 5
BATCH_SIZE = 16
MAX_LENGTH = 512
SEED = 42
DEBUG_INTERVAL = 100
SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/uncertainty_weighting"

set_seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

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
    lora_rank=32,           # increased rank for more capacity
    lora_alpha=32,
    lora_dropout=0.1,
    freeze_backbone=True,   # freeze except top layers
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

optimizer = torch.optim.Adam(
    list(backbone.parameters()) + list(engine.task_heads.parameters()) + [engine.log_vars],
    lr=1e-3
)

def safe_history(history_dict):
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
    print("\n[SANITY CHECK] Forward + backward pass for 1 batch per task")

    avg_loss = engine.train_one_epoch(optimizer, max_batches_per_loader=1)
    val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=0)

    log_vars_safe = [float(lv) for lv in engine.log_vars]
    grad_norms_safe = [float(x) if isinstance(x,float) else int(x) for x in engine.gradient_norms_log[-1]]

    # Save temporary history
    engine.history["train_loss"].append(float(avg_loss))
    engine.history["val_metrics"].append(val_metrics)
    engine.history["log_vars"].append(log_vars_safe)
    engine.history["gradient_norms"].append(grad_norms_safe)

    sanity_history_path = os.path.join(SAVE_DIR, "sanity_history.json")
    with open(sanity_history_path, 'w') as f:
        json.dump(safe_history(engine.history), f, indent=2)
    print(f"[SANITY] History saved successfully to {sanity_history_path}")

    sanity_model_path = os.path.join(SAVE_DIR, "sanity_model.pt")
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'task_heads_state_dict': engine.task_heads.state_dict(),
        'log_vars_state_dict': engine.log_vars.detach(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 0
    }, sanity_model_path)
    print(f"[SANITY] Model checkpoint saved to {sanity_model_path}")

    for task_idx, cfg in enumerate(task_configs):
        head_path = os.path.join(SAVE_DIR, f"sanity_head_task{task_idx}_{cfg['name']}.pt")
        torch.save(engine.task_heads[task_idx].state_dict(), head_path)
        print(f"[SANITY] Head for task {task_idx} saved to {head_path}")

    checkpoint = torch.load(sanity_model_path, map_location=DEVICE)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    engine.task_heads.load_state_dict(checkpoint['task_heads_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("[SANITY] Checkpoint loaded successfully. ✅")

    exit(0)

# Full training loop (only runs if SANITY_CHECK=False)
for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

    avg_loss = engine.train_one_epoch(optimizer)
    print(f"[Epoch {epoch+1}] Overall average training loss: {avg_loss:.4f}")

    val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=epoch)

    log_vars_safe = [float(lv) for lv in engine.log_vars]
    grad_norms_safe = [float(x) if isinstance(x,float) else int(x) for x in engine.gradient_norms_log[-1]]

    engine.history["train_loss"].append(float(avg_loss))
    engine.history["val_metrics"].append(val_metrics)
    engine.history["log_vars"].append(log_vars_safe)
    engine.history["gradient_norms"].append(grad_norms_safe)

    overall_path = os.path.join(SAVE_DIR, f"best_model_epoch{epoch+1}.pt")
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'task_heads_state_dict': engine.task_heads.state_dict(),
        'log_vars_state_dict': engine.log_vars.detach(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch+1
    }, overall_path)
    print(f"[INFO] Saved overall checkpoint to {overall_path}")

    for task_idx, cfg in enumerate(task_configs):
        improved = False
        if cfg['type'] == 'regression':
            metric_val = val_metrics[task_idx]['rmse']
            if engine.best["scores"][task_idx] == metric_val:
                improved = True
        else:
            metric_val = val_metrics[task_idx]['accuracy']
            if engine.best["scores"][task_idx] == metric_val:
                improved = True

        if improved:
            head_path = os.path.join(SAVE_DIR, f"best_head_task{task_idx}_{cfg['name']}.pt")
            torch.save(engine.task_heads[task_idx].state_dict(), head_path)
            print(f"[INFO] Saved best head for task {task_idx} ({cfg['name']}) to {head_path}")

    history_path = os.path.join(SAVE_DIR, f"history_epoch{epoch+1}.json")
    with open(history_path, 'w') as f:
        json.dump(safe_history(engine.history), f, indent=2)
    print(f"[INFO] Saved training history to {history_path}")

print("\n[TEST] Evaluating final model on test sets...")
test_metrics = engine.evaluate(engine.test_loaders, split_name="Test")
engine.history["test_metrics"] = test_metrics

final_metrics_path = os.path.join(SAVE_DIR, "final_metrics.json")
with open(final_metrics_path, 'w') as f:
    json.dump(safe_history(engine.history), f, indent=2)
print(f"[INFO] Saved final metrics and history to {final_metrics_path}")

engine.print_best_metrics()
