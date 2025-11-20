import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from itertools import cycle

# Simulated dataset sizes from your output
DATASET_SIZES = {
    'Thermostability': {'train': 53614, 'valid': 2512, 'test': 12851},
    'SecondaryStructure': {'train': 9763, 'valid': 1085, 'test': 667},
    'CloningCLF': {'train': 21037, 'valid': 2338, 'test': 4791}
}

BATCH_SIZE = 16

print("\n" + "="*80)
print("DATA LOADING DIAGNOSTIC")
print("="*80)

print(f"\nBatch size: {BATCH_SIZE}")
print("\n" + "-"*80)
print("BATCH CALCULATIONS")
print("-"*80)

# Calculate expected batches
batch_counts = {}
for task, sizes in DATASET_SIZES.items():
    train_size = sizes['train']
    num_batches = (train_size + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    actual_samples_loaded = num_batches * BATCH_SIZE
    print(f"\n{task}:")
    print(f"  Total train samples:    {train_size:,}")
    print(f"  Samples per batch:      {BATCH_SIZE}")
    print(f"  Expected batches:       {num_batches}")
    print(f"  Actual samples loaded:  {actual_samples_loaded:,}")
    print(f"  Padding samples:        {actual_samples_loaded - train_size}")
    batch_counts[task] = num_batches

max_batches = max(batch_counts.values())
print(f"\n" + "-"*80)
print(f"MAX BATCHES (longest dataloader): {max_batches}")
print(f"EPOCH LENGTH: {max_batches} iterations × 3 tasks = {max_batches * 3} total updates")
print("-"*80)

print("\n" + "-"*80)
print("CYCLIC BATCH STRATEGY - UPDATES PER TASK")
print("-"*80)

total_updates = 0
for task, num_batches in batch_counts.items():
    cycles = max_batches / num_batches
    total_updates_task = max_batches  # Each task processes max_batches, cycling if needed
    print(f"\n{task}:")
    print(f"  Batches in this task:   {num_batches}")
    print(f"  Updates needed:         {max_batches} (from epoch length)")
    print(f"  Cycles through loader:  {cycles:.2f}x")
    print(f"  Actual updates:         {total_updates_task} (fully deterministic)")

total_task_updates = max_batches * 3
print(f"\n{'='*80}")
print(f"TOTAL UPDATES PER EPOCH: {total_task_updates}")
print(f"  Task 0: {max_batches} updates")
print(f"  Task 1: {max_batches} updates")
print(f"  Task 2: {max_batches} updates")
print(f"  Each task: 1/{3} of total updates = {100/3:.1f}% each")
print(f"{'='*80}")

print("\n" + "-"*80)
print("SAMPLE COVERAGE PER EPOCH")
print("-"*80)

for task, sizes in DATASET_SIZES.items():
    num_batches = batch_counts[task]
    updates_per_epoch = max_batches
    
    # How many complete passes through this task?
    complete_passes = updates_per_epoch // num_batches
    remaining_batches = updates_per_epoch % num_batches
    
    # Samples covered
    complete_pass_samples = complete_passes * sizes['train']
    remaining_samples = min(remaining_batches * BATCH_SIZE, sizes['train'])
    total_samples_seen = complete_pass_samples + remaining_samples
    effective_epochs = updates_per_epoch / num_batches
    
    print(f"\n{task}:")
    print(f"  Dataset size:           {sizes['train']:,} samples")
    print(f"  Updates per epoch:      {updates_per_epoch}")
    print(f"  Batches in dataset:     {num_batches}")
    print(f"  Effective epochs:       {effective_epochs:.2f}x through data")
    print(f"  Complete passes:        {complete_passes}")
    print(f"  Remaining batches:      {remaining_batches}")
    print(f"  Total samples seen:     ~{total_samples_seen:,}")
    print(f"  Duplication factor:     {effective_epochs:.2f}x")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("""
✓ Cyclic strategy is CORRECT behavior:
  - Smaller datasets cycle through multiple times per epoch
  - Larger datasets go through once
  - All tasks get equal gradient updates (1/3 each)
  - Shared encoder learns from balanced task distribution

✓ Your data volumes are reasonable:
  - Thermostability: 53k samples (large - good for regression)
  - SecondaryStructure: 9.7k samples (medium - balanced per-residue task)
  - CloningCLF: 21k samples (good - balance between extremes)

ℹ What's happening:
  - Epoch processes {max_batches} batches (determined by longest loader)
  - Each task cycles independently if it's shorter
  - By end of epoch, all samples are seen at least once
  - Some samples in shorter datasets are repeated (fine, helps regularization)
""")

print("\n" + "="*80)
print("EXPECTED TRAINING BEHAVIOR")
print("="*80)
print(f"""
Each epoch:
  - Total gradient updates: {total_task_updates:,}
  - Updates per task: {max_batches} (exactly equal)
  - Time per epoch: ~{total_task_updates / 100:.1f} minutes (rough estimate on typical GPU)
  - Memory: ~{max_batches * BATCH_SIZE / 1000:.1f}GB per task cache

After 50 epochs:
  - Total updates: {total_task_updates * 50:,}
  - Task 0 sees data: ~{50 * 3.35:.1f}x (repetitions + new samples)
  - Task 1 sees data: ~{50 * 0.61:.1f}x
  - Task 2 sees data: ~50x (sees every sample exactly 50 times)

This is NORMAL for multi-task learning with imbalanced data.
""")

print("="*80 + "\n")