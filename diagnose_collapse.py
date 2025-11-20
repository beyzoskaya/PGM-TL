import torch
import torch.nn as nn
from protbert_hf import SharedProtBert
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from torch.utils.data import DataLoader
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 512

def collate_fn(batch, tokenizer, max_length=MAX_LENGTH):
    """Same as main script"""
    sequences = [item['sequence'] for item in batch]
    targets = [item['targets']['target'] for item in batch]

    encodings = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    if isinstance(targets[0], (int, float)):
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    else:
        max_len = input_ids.size(1)
        padded_targets = []
        for t in targets:
            t_tensor = torch.tensor(t, dtype=torch.long) if isinstance(t, list) else torch.tensor([t], dtype=torch.long)
            if len(t_tensor) < max_len:
                pad_len = max_len - len(t_tensor)
                t_tensor = torch.cat([t_tensor, torch.full((pad_len,), -100, dtype=torch.long)])
            else:
                t_tensor = t_tensor[:max_len]
            padded_targets.append(t_tensor)
        targets = torch.stack(padded_targets, dim=0)

    return {
        'sequence': input_ids,
        'attention_mask': attention_mask,
        'targets': {'target': targets}
    }

print("="*70)
print("DIAGNOSTIC: Why Classification Tasks Not Learning")
print("="*70)

# Load data
print("\n[1] Loading datasets...")
thermo = Thermostability(verbose=0)
thermo_train, _, _ = thermo.split()

ssp = SecondaryStructure(verbose=0)
ssp_train, _, _ = ssp.split()

clf = CloningCLF(verbose=0)
clf_train, _, _ = clf.split()

# Load model
print("[2] Loading ProtBERT...")
backbone = SharedProtBert(lora=True, lora_rank=16, verbose=False)
backbone = backbone.to(DEVICE)

# Create loaders
print("[3] Creating data loaders...")
tok = backbone.tokenizer

train_loader_0 = DataLoader(thermo_train, batch_size=2, shuffle=False,
                           collate_fn=lambda b: collate_fn(b, tok, MAX_LENGTH))
train_loader_1 = DataLoader(ssp_train, batch_size=2, shuffle=False,
                           collate_fn=lambda b: collate_fn(b, tok, MAX_LENGTH))
train_loader_2 = DataLoader(clf_train, batch_size=2, shuffle=False,
                           collate_fn=lambda b: collate_fn(b, tok, MAX_LENGTH))

# Build heads
from protbert_hf import build_regression_head, build_token_classification_head, build_sequence_classification_head
hidden_dim = backbone.hidden_size
head_0 = build_regression_head(hidden_dim, 1).to(DEVICE)
head_1 = build_token_classification_head(hidden_dim, 8).to(DEVICE)
head_2 = build_sequence_classification_head(hidden_dim, 2).to(DEVICE)

loss_fn_0 = nn.MSELoss()
loss_fn_1 = nn.CrossEntropyLoss(ignore_index=-100)
loss_fn_2 = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(backbone.parameters()) + 
    list(head_0.parameters()) +
    list(head_1.parameters()) +
    list(head_2.parameters()),
    lr=5e-4
)

# TEST 1: Task 0 (Regression) - Should work
print("\n" + "="*70)
print("TEST 1: Task 0 (Regression) - Check if gradients flow")
print("="*70)

backbone.train()
head_0.train()

batch = next(iter(train_loader_0))
input_ids = batch['sequence'].to(DEVICE)
attention_mask = batch['attention_mask'].to(DEVICE)
targets = batch['targets']['target'].to(DEVICE)

# Save initial weights
init_weight_0 = head_0[3].weight.data.clone()

optimizer.zero_grad()
embeddings = backbone(input_ids, attention_mask, per_residue=False)
logits = head_0(embeddings)
loss = loss_fn_0(logits.squeeze(-1), targets.squeeze(-1))

print(f"Input shape: {input_ids.shape}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Loss: {loss.item():.6f}")

loss.backward()

# Check gradients
has_grad = False
for name, p in backbone.named_parameters():
    if p.grad is not None and p.grad.abs().sum() > 0:
        has_grad = True
        print(f"✓ Backbone has gradient in {name}: norm={p.grad.norm().item():.6e}")
        break

if not has_grad:
    print("❌ WARNING: Backbone has NO gradients for Task 0!")

head_grad = head_0[3].weight.grad
if head_grad is not None and head_grad.abs().sum() > 0:
    print(f"✓ Head 0 has gradient: norm={head_grad.norm().item():.6e}")
else:
    print("❌ WARNING: Head 0 has NO gradients!")

optimizer.step()

# Check if weights changed
weight_change = (head_0[3].weight.data - init_weight_0).abs().sum().item()
print(f"Head 0 weight change after step: {weight_change:.6e}")
if weight_change < 1e-8:
    print("⚠️  WARNING: Head weights NOT updating!")

# TEST 2: Task 1 (Per-residue Classification)
print("\n" + "="*70)
print("TEST 2: Task 1 (Per-residue) - Check if gradients flow")
print("="*70)

backbone.train()
head_1.train()

batch = next(iter(train_loader_1))
input_ids = batch['sequence'].to(DEVICE)
attention_mask = batch['attention_mask'].to(DEVICE)
targets = batch['targets']['target'].to(DEVICE)

print(f"Input shape: {input_ids.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Targets dtype: {targets.dtype}")

# CRITICAL: Check per_residue flag
per_residue = True  # This should be True for Task 1
embeddings = backbone(input_ids, attention_mask, per_residue=per_residue)
print(f"Embeddings shape (per_residue={per_residue}): {embeddings.shape}")

logits = head_1(embeddings)
print(f"Logits shape: {logits.shape}")

# Flatten for loss
B, L, C = logits.shape
logits_flat = logits.view(B*L, C)
targets_flat = targets.view(B*L)
print(f"Logits flat: {logits_flat.shape}")
print(f"Targets flat: {targets_flat.shape}")

loss = loss_fn_1(logits_flat, targets_flat)
print(f"Loss: {loss.item():.6f}")

optimizer.zero_grad()
loss.backward()

# Check gradients
has_grad = False
for name, p in backbone.named_parameters():
    if p.grad is not None and p.grad.abs().sum() > 0:
        has_grad = True
        print(f"✓ Backbone has gradient in {name}: norm={p.grad.norm().item():.6e}")
        break

if not has_grad:
    print("❌ CRITICAL: Backbone has NO gradients for Task 1!")
    print("   This is why accuracy is frozen!")

head_grad = head_1[2].weight.grad
if head_grad is not None:
    print(f"✓ Head 1 has gradient: norm={head_grad.norm().item():.6e}")
else:
    print("❌ WARNING: Head 1 has NO gradients!")

# TEST 3: Task 2 (Sequence Classification)
print("\n" + "="*70)
print("TEST 3: Task 2 (Sequence) - Check if gradients flow")
print("="*70)

backbone.train()
head_2.train()

batch = next(iter(train_loader_2))
input_ids = batch['sequence'].to(DEVICE)
attention_mask = batch['attention_mask'].to(DEVICE)
targets = batch['targets']['target'].to(DEVICE)

print(f"Input shape: {input_ids.shape}")
print(f"Targets shape: {targets.shape}")

# CRITICAL: per_residue should be False for Task 2
per_residue = False
embeddings = backbone(input_ids, attention_mask, per_residue=per_residue)
print(f"Embeddings shape (per_residue={per_residue}): {embeddings.shape}")

logits = head_2(embeddings)
print(f"Logits shape: {logits.shape}")

targets_flat = targets.view(-1)
loss = loss_fn_2(logits, targets_flat.long())
print(f"Loss: {loss.item():.6f}")

optimizer.zero_grad()
loss.backward()

# Check gradients
has_grad = False
for name, p in backbone.named_parameters():
    if p.grad is not None and p.grad.abs().sum() > 0:
        has_grad = True
        print(f"✓ Backbone has gradient in {name}: norm={p.grad.norm().item():.6e}")
        break

if not has_grad:
    print("❌ CRITICAL: Backbone has NO gradients for Task 2!")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
If Task 1 and Task 2 show "CRITICAL: Backbone has NO gradients":
  → Problem is in forward pass or embedding computation
  → Check if per_residue flag is correctly passed
  → Check if embeddings are actually connected to input_ids
  
If Task 0 shows "✓ Backbone has gradient" but others don't:
  → The backbone forward pass is task-dependent
  → Something in per_residue logic is breaking gradients
""")