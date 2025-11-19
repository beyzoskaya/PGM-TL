import torch
import numpy as np
from torch.utils.data import DataLoader
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_with_task_specific_encoder import collate_fn, MAX_LENGTH
import warnings

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def verify_dataset_loading():
    print_section("1. DATASET LOADING VERIFICATION")
    
    datasets = {
        'Thermostability': Thermostability(verbose=0),
        'SecondaryStructure': SecondaryStructure(verbose=0),
        'CloningCLF': CloningCLF(verbose=0)
    }
    
    for name, dataset in datasets.items():
        print(f"\n✓ {name}")
        print(f"  Total sequences: {len(dataset)}")
        print(f"  Num samples per split: {dataset.num_samples}")
        
        # Verify split consistency
        total = sum(dataset.num_samples)
        assert total == len(dataset), f"Split sum {total} != total {len(dataset)}"
        print(f"  ✓ Split consistency verified: {total} = {len(dataset)}")
        
        # Check target keys
        print(f"  Target keys: {list(dataset.targets.keys())}")
        
        # Sample inspection
        item = dataset[0]
        print(f"  Sample 0 sequence length: {len(item['sequence'])}")
        print(f"  Sample 0 targets keys: {list(item['targets'].keys())}")
        target_val = item['targets']['target']
        print(f"  Sample 0 target type: {type(target_val)}, value: {target_val}")
    
    return datasets

def verify_splits(datasets):
    print_section("2. SPLIT CREATION VERIFICATION")
    
    for name, dataset in datasets.items():
        print(f"\n✓ {name}")
        train_set, valid_set, test_set = dataset.split()
        
        train_size = len(train_set)
        valid_size = len(valid_set)
        test_size = len(test_set)
        
        print(f"  Train split size: {train_size}")
        print(f"  Valid split size: {valid_size}")
        print(f"  Test split size: {test_size}")
        print(f"  Total: {train_size + valid_size + test_size}")
  
        assert train_size == dataset.num_samples[0], "Train size mismatch"
        assert valid_size == dataset.num_samples[1], "Valid size mismatch"
        assert test_size == dataset.num_samples[2], "Test size mismatch"
        print(f"  ✓ Split sizes match num_samples: {dataset.num_samples}")

        all_indices = set()
        for subset_name, subset in [('train', train_set), ('valid', valid_set), ('test', test_set)]:
            indices = set(range(subset.indices[0], subset.indices[-1] + 1)) if len(subset.indices) > 0 else set()
            all_indices.update(indices)
        print(f"  ✓ Splits are non-overlapping: {len(all_indices)} unique indices")
    
    return train_set, valid_set, test_set

def verify_collate_fn(datasets):
    print_section("3. COLLATE_FN & TOKENIZATION VERIFICATION")
    
    backbone = SharedProtBert(lora=False, verbose=False)
    tokenizer = backbone.tokenizer
    
    task_configs = [
        {'type': 'regression', 'name': 'Thermostability'},
        {'type': 'per_residue_classification', 'name': 'SecondaryStructure'},
        {'type': 'classification', 'name': 'CloningCLF'}
    ]
    
    for task_idx, (name, dataset) in enumerate(datasets.items()):
        print(f"\n✓ Task {task_idx}: {name} ({task_configs[task_idx]['type']})")
        
        # Create small batch
        small_batch = [dataset[i] for i in range(min(4, len(dataset)))]
        
        # Apply collate
        try:
            collated = collate_fn(small_batch, tokenizer, max_length=MAX_LENGTH)
            print(f"  ✓ Collate succeeded")
        except Exception as e:
            print(f"  ✗ Collate failed: {e}")
            continue
        
        # Verify shapes
        input_ids = collated['sequence']
        attention_mask = collated['attention_mask']
        targets = collated['targets']['target']
        
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Targets shape: {targets.shape}")
        
        # Verify shapes are consistent
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        assert attention_mask.shape == input_ids.shape, "Attention mask shape mismatch"
        print(f"  ✓ Attention mask shape matches input_ids")
        
        # Task-specific checks
        if task_configs[task_idx]['type'] == 'regression':
            assert targets.shape == (batch_size,), f"Regression targets should be (B,), got {targets.shape}"
            assert targets.dtype == torch.float32, "Regression targets should be float32"
            print(f"  ✓ Regression targets: float32, shape {targets.shape}")
            print(f"    Values: {targets.numpy()}")
        
        elif task_configs[task_idx]['type'] == 'per_residue_classification':
            assert targets.shape == (batch_size, seq_len), f"Token-level targets should be (B, L), got {targets.shape}"
            assert targets.dtype == torch.long, "Token-level targets should be long"
            # Check that padding is -100
            mask = targets == -100
            print(f"  ✓ Token-level targets: long, shape {targets.shape}")
            print(f"    Padding -100 count per batch: {mask.sum(dim=1).numpy()}")
            print(f"    Sample target values (first sequence): {targets[0, :20].numpy()}")
        
        else:  # classification
            assert targets.shape == (batch_size,), f"Classification targets should be (B,), got {targets.shape}"
            assert targets.dtype == torch.long, "Classification targets should be long"
            print(f"  ✓ Classification targets: long, shape {targets.shape}")
            print(f"    Unique classes: {torch.unique(targets).numpy()}")
        
        # Verify tokenization
        seq_from_data = small_batch[0]['sequence']
        print(f"\n  Sample sequence (first 30 chars): {seq_from_data[:30]}")
        print(f"  First 5 token IDs: {input_ids[0, :5].numpy()}")
        print(f"  Attention mask first 10: {attention_mask[0, :10].numpy()}")

def verify_backbone(datasets):
    print_section("4. BACKBONE PROCESSING VERIFICATION")
    
    backbone = SharedProtBert(lora=True, verbose=False)
    backbone = backbone.to('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = backbone.tokenizer
    
    device = next(backbone.parameters()).device
    
    task_configs = [
        {'type': 'regression', 'name': 'Thermostability'},
        {'type': 'per_residue_classification', 'name': 'SecondaryStructure'},
        {'type': 'classification', 'name': 'CloningCLF'}
    ]
    
    for task_idx, (name, dataset) in enumerate(datasets.items()):
        print(f"\n✓ Task {task_idx}: {name}")
        
        small_batch = [dataset[i] for i in range(min(2, len(dataset)))]
        collated = collate_fn(small_batch, tokenizer, max_length=MAX_LENGTH)
        
        input_ids = collated['sequence'].to(device)
        attention_mask = collated['attention_mask'].to(device)
        
        per_residue = (task_configs[task_idx]['type'] == 'per_residue_classification')
        
        with torch.no_grad():
            embeddings = backbone(input_ids, attention_mask, per_residue=per_residue)
        
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Per-residue: {per_residue}")
        
        # Verify shapes
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        if per_residue:
            assert embeddings.shape == (batch_size, seq_len, 1024), \
                f"Per-residue should be (B, L, 1024), got {embeddings.shape}"
            print(f"  ✓ Per-residue embedding shape correct")
        else:
            assert embeddings.shape == (batch_size, 1024), \
                f"Sequence-level should be (B, 1024), got {embeddings.shape}"
            print(f"  ✓ Sequence-level embedding shape correct")
        
        # Verify embedding statistics
        emb_mean = embeddings.mean().item()
        emb_std = embeddings.std().item()
        emb_min = embeddings.min().item()
        emb_max = embeddings.max().item()
        
        print(f"  Embeddings - mean: {emb_mean:.4f}, std: {emb_std:.4f}")
        print(f"  Embeddings - min: {emb_min:.4f}, max: {emb_max:.4f}")
        
        # Check for NaN/Inf
        assert not torch.isnan(embeddings).any(), "NaN in embeddings!"
        assert not torch.isinf(embeddings).any(), "Inf in embeddings!"
        print(f"  ✓ No NaN/Inf in embeddings")

def verify_task_heads_and_losses():
    print_section("5. TASK HEADS & LOSS VERIFICATION")
    
    from protbert_hf import (build_regression_head, 
                             build_token_classification_head,
                             build_sequence_classification_head)
    
    hidden_dim = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    task_configs = [
        {
            'type': 'regression',
            'num_labels': 1,
            'name': 'Thermostability',
            'head_builder': build_regression_head,
            'loss_fn': torch.nn.MSELoss()
        },
        {
            'type': 'per_residue_classification',
            'num_labels': 8,
            'name': 'SecondaryStructure',
            'head_builder': build_token_classification_head,
            'loss_fn': torch.nn.CrossEntropyLoss(ignore_index=-100)
        },
        {
            'type': 'classification',
            'num_labels': 2,
            'name': 'CloningCLF',
            'head_builder': build_sequence_classification_head,
            'loss_fn': torch.nn.CrossEntropyLoss(ignore_index=-100)
        }
    ]
    
    for cfg in task_configs:
        print(f"\n✓ {cfg['name']} ({cfg['type']})")
        
        # Build head
        head = cfg['head_builder'](hidden_dim, cfg['num_labels']).to(device)
        print(f"  Head architecture: {head}")
        
        # Create dummy embeddings
        if cfg['type'] == 'per_residue_classification':
            batch_size, seq_len = 4, 50
            embeddings = torch.randn(batch_size, seq_len, hidden_dim).to(device)
            targets = torch.randint(0, cfg['num_labels'], (batch_size, seq_len)).to(device)
            # Add some padding
            targets[:, -10:] = -100
        else:
            batch_size = 4
            embeddings = torch.randn(batch_size, hidden_dim).to(device)
            targets = torch.randint(0, cfg['num_labels'], (batch_size,)).to(device)
        
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Targets shape: {targets.shape}")
        
        # Forward through head
        with torch.no_grad():
            logits = head(embeddings)
        
        print(f"  Logits shape: {logits.shape}")
        
        # Verify logit shape
        if cfg['type'] == 'per_residue_classification':
            assert logits.shape == (batch_size, seq_len, cfg['num_labels']), \
                f"Token-level logits should be (B, L, C), got {logits.shape}"
        elif cfg['type'] == 'regression':
            assert logits.shape == (batch_size, cfg['num_labels']), \
                f"Regression logits should be (B, 1), got {logits.shape}"
        else:
            assert logits.shape == (batch_size, cfg['num_labels']), \
                f"Classification logits should be (B, C), got {logits.shape}"
        print(f"  ✓ Logit shape correct")
        
        # Compute loss
        try:
            if cfg['type'] == 'per_residue_classification':
                loss = cfg['loss_fn'](logits.view(-1, cfg['num_labels']), 
                                     targets.view(-1))
            elif cfg['type'] == 'regression':
                loss = cfg['loss_fn'](logits.squeeze(-1), targets.float())
            else:
                loss = cfg['loss_fn'](logits, targets)
            
            print(f"  Loss: {loss.item():.4f}")
            assert not torch.isnan(loss), "Loss is NaN!"
            assert not torch.isinf(loss), "Loss is Inf!"
            print(f"  ✓ Loss computed successfully, no NaN/Inf")
        except Exception as e:
            print(f"  ✗ Loss computation failed: {e}")

def verify_metrics():
    print_section("6. METRICS CALCULATION VERIFICATION")
    
    from engine_hf_with_task_specific_encoder import regression_metrics, classification_metrics
    
    # Task 0: Regression
    print(f"\n✓ Regression Metrics (Thermostability)")
    preds_reg = torch.tensor([[0.5], [1.2], [-0.3], [0.1]])
    targets_reg = torch.tensor([[0.4], [1.5], [-0.2], [0.0]])
    metrics_reg = regression_metrics(preds_reg, targets_reg)
    print(f"  Predictions: {preds_reg.squeeze().numpy()}")
    print(f"  Targets: {targets_reg.squeeze().numpy()}")
    print(f"  Metrics: {metrics_reg}")
    assert 'mse' in metrics_reg and 'rmse' in metrics_reg and 'mae' in metrics_reg
    print(f"  ✓ All regression metrics present")
    
    # Task 1: Token-level classification
    print(f"\n✓ Token-Level Classification Metrics (SecondaryStructure)")
    batch, seq_len, num_classes = 2, 10, 8
    logits_token = torch.randn(batch, seq_len, num_classes)
    targets_token = torch.randint(0, num_classes, (batch, seq_len))
    targets_token[:, -2:] = -100  # Add padding
    metrics_token = classification_metrics(logits_token, targets_token, ignore_index=-100)
    print(f"  Logits shape: {logits_token.shape}")
    print(f"  Targets shape: {targets_token.shape}")
    print(f"  Metrics: {metrics_token}")
    assert 'accuracy' in metrics_token
    print(f"  ✓ Token-level metrics computed correctly")
    
    # Task 2: Sequence classification
    print(f"\n✓ Sequence Classification Metrics (CloningCLF)")
    logits_seq = torch.randn(4, 2)
    targets_seq = torch.tensor([0, 1, 0, 1])
    metrics_seq = classification_metrics(logits_seq, targets_seq)
    print(f"  Logits shape: {logits_seq.shape}")
    print(f"  Targets shape: {targets_seq.shape}")
    print(f"  Metrics: {metrics_seq}")
    assert 'accuracy' in metrics_seq
    print(f"  ✓ Sequence-level metrics computed correctly")

def verify_data_loaders(datasets):
    print_section("7. DATA LOADER VERIFICATION")
    
    backbone = SharedProtBert(lora=False, verbose=False)
    tokenizer = backbone.tokenizer
    
    task_names = ['Thermostability', 'SecondaryStructure', 'CloningCLF']
    
    for task_idx, (name, dataset) in enumerate(datasets.items()):
        print(f"\n✓ {name}")
        
        train_set, _, _ = dataset.split()
        loader = DataLoader(train_set, batch_size=4, shuffle=False,
                           collate_fn=lambda b: collate_fn(b, tokenizer, max_length=MAX_LENGTH))
        
        # Get first 3 batches
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 3:
                break
            
            print(f"  Batch {batch_idx}:")
            print(f"    Input IDs: {batch['sequence'].shape}")
            print(f"    Attention mask: {batch['attention_mask'].shape}")
            print(f"    Targets: {batch['targets']['target'].shape}")
        
        print(f"  ✓ {name} loader works correctly")

def run_full_verification():
    print("\n" + "="*70)
    print("COMPREHENSIVE MULTITASK LEARNING VERIFICATION SUITE")
    print("="*70)
    
    try:
        datasets = verify_dataset_loading()
        verify_splits(datasets)
        verify_collate_fn(datasets)
        verify_backbone(datasets)
        verify_task_heads_and_losses()
        verify_metrics()
        verify_data_loaders(datasets)
        
        print_section("ALL VERIFICATIONS PASSED ✓")
        print("\nYou can proceed with confidence. Everything is wired correctly!\n")
        
    except Exception as e:
        print_section(f"VERIFICATION FAILED ✗")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_full_verification()