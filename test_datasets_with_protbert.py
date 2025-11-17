import torch
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert

def test_dataset(dataset_class, name):
    print(f"\n=== Testing {name} Dataset ===")
    dataset = dataset_class(path="./data")
    train_set, valid_set, test_set = dataset.split()
    
    print(f"Number of samples - Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")
    
    sample = next(iter(train_set))
    seq = sample.get('sequence', None)
    targs = sample.get('targets', {})
    
    if seq is None or not isinstance(targs, dict):
        raise ValueError(f"[{name}] Dataset must return dict with 'sequence' and 'targets'")
    
    print(f"Example sequence length: {len(seq)}")
    print(f"Target keys: {list(targs.keys())}")
    for k, v in targs.items():
        if isinstance(v, list):
            print(f"Example target for key '{k}': {v[:min(10, len(v))]} ...")
        else:
            print(f"Example target for key '{k}': {v}")
    
    return train_set, valid_set, test_set

def test_model(train_samples):
    print("\n=== Testing SharedProtBert Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model = SharedProtBert(lora=True)
    model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Take a small batch for forward pass
    batch_seqs = [s['sequence'] for s in train_samples[:2]]  # first 2 sequences
    enc = model.backbone.tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)

    for k, v in enc.items():
        enc[k] = v.to(device)
    
    with torch.no_grad():
        embeddings = model(enc['input_ids'], enc['attention_mask'])
    
    print("Forward pass successful!")
    print("Embeddings shape:", embeddings.shape)

if __name__ == "__main__":
    # Step 1: Test all datasets
    train_thermo, _, _ = test_dataset(Thermostability, "Thermostability")
    train_ss, _, _ = test_dataset(SecondaryStructure, "SecondaryStructure")
    train_clf, _, _ = test_dataset(CloningCLF, "CloningCLF")
    
    # Step 2: Test model with small batch from Thermostability
    test_model(train_thermo)
    
    # Optional: test forward pass with sequences from other datasets
    test_model(train_ss)
    test_model(train_clf)
    
    print("\n✅ All dataset + model checks completed successfully!")
