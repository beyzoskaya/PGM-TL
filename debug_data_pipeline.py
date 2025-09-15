# debug_data_pipeline.py
from datasets import load_dataset
from flip_hf import Thermostability
import torch

print("=== STEP 1: Raw HuggingFace Data ===")
dataset = load_dataset("proteinglm/stability_prediction")
sample = dataset['train'][0]
print(f"Raw sample: {sample}")
print(f"Raw label type: {type(sample['label'])}")
print(f"Raw label value: {sample['label']}")

print("\n=== STEP 2: Our Dataset Class ===")
thermo_dataset = Thermostability(path='./data', verbose=0)
print(f"Our dataset sequences length: {len(thermo_dataset.sequences)}")
print(f"Our dataset targets keys: {list(thermo_dataset.targets.keys())}")
print(f"First few target values: {thermo_dataset.targets['label'][:5]}")
print(f"Target type: {type(thermo_dataset.targets['label'][0])}")

print("\n=== STEP 3: Dataset __getitem__ ===")
item = thermo_dataset[0]
print(f"Item keys: {list(item.keys())}")
print(f"Item targets: {item['targets']}")
print(f"Item target types: {[(k, type(v)) for k, v in item['targets'].items()]}")

print("\n=== STEP 4: DataLoader collate_fn ===")
from torch.utils.data import DataLoader
from engine_hf import MultiTaskEngine

# Create dummy engine to get collate_fn
dummy_engine = MultiTaskEngine(
    tasks=[None], train_sets=[[]], valid_sets=[[]], test_sets=[[]],
    optimizer=None, batch_size=1
)

# Test collate function
batch_items = [thermo_dataset[0], thermo_dataset[1]]
print(f"Before collate - item 0 targets: {batch_items[0]['targets']}")
print(f"Before collate - item 1 targets: {batch_items[1]['targets']}")

collated = dummy_engine.collate_fn(batch_items)
print(f"After collate - targets: {collated['targets']}")
print(f"After collate - target types: {[(k, type(v)) for k, v in collated['targets'].items()]}")