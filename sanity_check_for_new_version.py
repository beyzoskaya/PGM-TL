import torch
from torch.utils.data import DataLoader
from flip_hf import create_dataset

BATCH_SIZE = 2  
DATASETS_TO_TEST = [
    "Thermostability",
    "SecondaryStructure",
    "BindingAffinityRegression"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

for dataset_name in DATASETS_TO_TEST:
    print(f"\n=== Testing dataset: {dataset_name} ===")
    
    # Create dataset
    dataset = create_dataset(dataset_name, verbose=1)
    
    # Split (train/valid/test)
    splits = dataset.split()
    train_split = splits[0]  # only test train split for sanity
    
    # Small subset for sanity check
    small_subset = torch.utils.data.Subset(train_split, range(min(2, len(train_split))))
    loader = DataLoader(small_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    for batch in loader:
        # batch is a list of dicts; sequences/targets/graph to tensors
        sequences = [item['sequence'] for item in batch]
        targets = [item['targets']['target'] for item in batch]
        graphs = [item['graph'] for item in batch]
        
        print("Sequences:", sequences)
        print("Targets:", targets)
        print("Graph residue types:", [g['residue_type'] for g in graphs])
        print("Graph num residues:", [g['num_residues'] for g in graphs])
        break  # only first batch for sanity
