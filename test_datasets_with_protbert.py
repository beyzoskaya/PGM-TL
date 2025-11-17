# test_integration.py
import torch
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert

def test_dataset(dataset_class, dataset_name, max_samples=3):
    print(f"\n=== Testing {dataset_name} Dataset ===")
    dataset = dataset_class(path="./data")  # adjust path if needed
    train_set, valid_set, test_set = dataset.split()
    
    print(f"Total samples - Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")
    
    # Take a few samples for sanity check
    for i in range(min(max_samples, len(train_set))):
        sample = train_set[i]
        seq = sample['sequence']
        targets = sample['targets']
        print(f"Sample {i}: sequence length={len(seq)}, target keys={list(targets.keys())}")
        for key in targets:
            val = targets[key]
            if isinstance(val, list):
                print(f"  Target '{key}' (first 10): {val[:10]}")
            else:
                print(f"  Target '{key}': {val}")
    
    return train_set, valid_set, test_set

def test_model(dataset_samples, model_name="SharedProtBert"):
    print(f"\n=== Testing {model_name} Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize shared ProtBert with LoRA enabled
    model = SharedProtBert(lora=True)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Take a small batch for forward pass
    batch_samples = [dataset_samples[i] for i in range(min(2, len(dataset_samples)))]
    batch_seqs = [s['sequence'] for s in batch_samples]
    
    # Tokenize
    enc = model.backbone.tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
    for k, v in enc.items():
        enc[k] = v.to(device)
    
    # Forward pass
    with torch.no_grad():
        embeddings = model(enc['input_ids'], enc['attention_mask'])
    
    print("Forward pass successful!")
    print("Embeddings shape:", embeddings.shape)

if __name__ == "__main__":
    torch.manual_seed(0)
    
    # Test Thermostability
    train_thermo, _, _ = test_dataset(Thermostability, "Thermostability")
    test_model(train_thermo)
    
    # Test SecondaryStructure
    train_ss, _, _ = test_dataset(SecondaryStructure, "SecondaryStructure")
    test_model(train_ss)
    
    # Test CloningCLF
    train_clone, _, _ = test_dataset(CloningCLF, "CloningCLF")
    test_model(train_clone)
    
    print("\n✅ All datasets and model forward passes checked successfully!")
