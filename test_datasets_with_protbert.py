from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
import torch

# Load one batch from each dataset
datasets = [
    Thermostability(path="./data"),
    SecondaryStructure(path="./data"),
    CloningCLF(path="./data")
]

for ds in datasets:
    train, valid, test = ds.split()
    sample = next(iter(train))
    print("\n--- Dataset Check ---")
    print("Sequence length:", len(sample['sequence']))
    print("Target keys:", sample['targets'].keys())
    print("Example target:", {k: sample['targets'][k] for k in sample['targets'].keys()})

model = SharedProtBert(lora=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

sample_seq = [sample['sequence'] for sample in train[:2]]  # take 2 sequences
enc = model.backbone.tokenizer(sample_seq, return_tensors="pt", padding=True, truncation=True)
for k, v in enc.items():
    enc[k] = v.to(device)

with torch.no_grad():
    embeddings = model(enc['input_ids'], enc['attention_mask'])
print("Embeddings shape:", embeddings.shape)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
