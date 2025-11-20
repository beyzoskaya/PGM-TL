# import torch
# from protbert_hf import SharedProtBert

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Device: {device}")

# # Create model
# model = SharedProtBert(lora=True, verbose=True).to(device)

# # Dummy input
# sequences = ["MKTWVTFISLLFLFSS", "ACDEFGHIKLMNPQRSTVWY"]
# tokenizer = model.tokenizer
# enc = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
# input_ids = enc["input_ids"].to(device)
# attention_mask = enc["attention_mask"].to(device)

# # Forward pass - pooled
# pooled = model(input_ids, attention_mask, per_residue=False)
# print(f"Pooled embeddings shape: {pooled.shape}")

# # Forward pass - per token
# tokens = model(input_ids, attention_mask, per_residue=True)
# print(f"Per-token embeddings shape: {tokens.shape}")


import torch
from torch import optim
from protbert_hf import SharedProtBert
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from engine_hf_with_task_specific_encoder import MultiTaskEngine, set_seed, MAX_LENGTH, ensure_dir

# ----------------------------
# Settings
# ----------------------------
SEED = 42
set_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = ensure_dir("./sanity_check_outputs")
BATCH_SIZE = 2  # small batch for quick test

print(f"Device: {device}")
print(f"Max sequence length: {MAX_LENGTH}")

# ----------------------------
# Load datasets
# ----------------------------
print("\n[DATA] Loading datasets...")
thermo_full = Thermostability(verbose=1)
thermo_train, thermo_valid, thermo_test = thermo_full.split()

ssp_full = SecondaryStructure(verbose=1)
ssp_train, ssp_valid, ssp_test = ssp_full.split()

clf_full = CloningCLF(verbose=1)
clf_train, clf_valid, clf_test = clf_full.split()

print("\n[DATA] Split sizes:")
print(f"  Thermostability: train={len(thermo_train)}, valid={len(thermo_valid)}, test={len(thermo_test)}")
print(f"  SecondaryStructure: train={len(ssp_train)}, valid={len(ssp_valid)}, test={len(ssp_test)}")
print(f"  CloningCLF: train={len(clf_train)}, valid={len(clf_valid)}, test={len(clf_test)}")

# ----------------------------
# Task configuration
# ----------------------------
task_configs = [
    {'type': 'regression', 'num_labels': 1, 'name': 'Thermostability'},
    {'type': 'per_residue_classification', 'num_labels': 8, 'name': 'SecondaryStructure'},
    {'type': 'classification', 'num_labels': 2, 'name': 'CloningCLF'}
]

# ----------------------------
# Initialize PEFT ProtBERT
# ----------------------------
print("\n[MODEL] Initializing shared ProtBERT with LoRA (PEFT)...")
backbone = SharedProtBert(lora=True, verbose=True).to(device)

# Quick parameter check
trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"Trainable param count: {trainable_params}")
for name, p in backbone.named_parameters():
    if p.requires_grad:
        print(f"  Trainable: {name} -> {tuple(p.shape)}")

# ----------------------------
# Initialize MultiTaskEngine
# ----------------------------
engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=task_configs,
    train_sets=[thermo_train, ssp_train, clf_train],
    valid_sets=[thermo_valid, ssp_valid, clf_valid],
    test_sets=[thermo_test, ssp_test, clf_test],
    batch_size=BATCH_SIZE,
    device=device,
    verbose=True,
    grad_clip=1.0
)

# ----------------------------
# Optimizer (LoRA params only)
# ----------------------------
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, list(backbone.parameters()) + list(engine.task_heads.parameters())),
    lr=2e-4,
    weight_decay=1e-5
)

# ----------------------------
# Sanity check: single forward + backward
# ----------------------------
print("\n[SANITY CHECK] Forward + backward pass for one batch per task")
for task_idx, loader in enumerate(engine.train_loaders):
    batch = next(iter(loader))
    input_ids = batch['sequence'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    targets = batch['targets']['target'].to(device)

    optimizer.zero_grad()
    per_residue = (task_configs[task_idx]['type'] == 'per_residue_classification')
    embeddings = backbone(input_ids, attention_mask, per_residue=per_residue)
    print(f"[Task {task_idx}] Embeddings shape: {embeddings.shape}")

    logits = engine.forward_task(embeddings, task_idx)
    print(f"[Task {task_idx}] Logits shape: {logits.shape}")

    loss = engine.compute_loss(logits, targets, task_idx)
    print(f"[Task {task_idx}] Loss: {loss.item():.4f}")

    loss.backward()
    print(f"[Task {task_idx}] Backward pass done")

# ----------------------------
# Test training one mini-epoch
# ----------------------------
print("\n[SANITY CHECK] Training one mini-epoch (2 batches per task max)")
train_loss = engine.train_one_epoch(optimizer, max_batches_per_loader=2)
print(f"Mini-epoch avg loss: {train_loss:.4f}")

# ----------------------------
# Test evaluation
# ----------------------------
print("\n[SANITY CHECK] Evaluating validation set")
val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=0)
print("Validation metrics:", val_metrics)

engine.print_best_metrics()
print("\n✓ Sanity check completed successfully")
