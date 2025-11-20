import torch
from protbert_hf import SharedProtBert

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create model
model = SharedProtBert(lora=True, verbose=True).to(device)

# Dummy input
sequences = ["MKTWVTFISLLFLFSS", "ACDEFGHIKLMNPQRSTVWY"]
tokenizer = model.tokenizer
enc = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)

# Forward pass - pooled
pooled = model(input_ids, attention_mask, per_residue=False)
print(f"Pooled embeddings shape: {pooled.shape}")

# Forward pass - per token
tokens = model(input_ids, attention_mask, per_residue=True)
print(f"Per-token embeddings shape: {tokens.shape}")
