from flip_hf import PeptideHLAMHCAffinity
import torch
import numpy as np

dataset = PeptideHLAMHCAffinity(path='./data')
train_set, _, _ = dataset.split()
print(f"Full dataset: {len(train_set)} samples")

labels = []
for i in range(len(train_set)):
    label = train_set[i]['targets']['label']
    labels.append(label)

labels = np.array(labels)
unique, counts = np.unique(labels, return_counts=True)
balance = min(counts) / max(counts)

print(f"Full dataset distribution: {unique} -> {counts}")
print(f"Balance score: {balance:.3f} (1.0=perfect, <0.3=problematic)")

print(f"\nTesting random 15k sampling:")
for seed in [0, 1, 2]:
    torch.manual_seed(seed)
    indices = torch.randperm(len(train_set))[:15000]
    
    sample_labels = labels[indices]
    unique_s, counts_s = np.unique(sample_labels, return_counts=True)
    balance_s = min(counts_s) / max(counts_s)
    
    print(f"Seed {seed}: {counts_s} (balance: {balance_s:.3f})")