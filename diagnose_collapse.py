import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from protbert_hf import SharedProtBert
from flip_hf import Thermostability, SecondaryStructure, CloningCLF

def diagnose_trainable_params(model):
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: TRAINABLE PARAMETERS")
    print("="*70)
    
    trainable = []
    frozen = []
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable.append((name, p.shape, p.numel()))
        else:
            frozen.append((name, p.shape))
    
    print(f"\n✓ Trainable parameters ({len(trainable)} components):")
    total_trainable = 0
    for name, shape, numel in trainable[:20]: 
        print(f"  {name}: {shape} ({numel:,} params)")
        total_trainable += numel
    if len(trainable) > 20:
        print(f"  ... and {len(trainable) - 20} more")
    
    print(f"\n⚠️  Total trainable: {total_trainable:,}")
    
    print(f"\n❌ Frozen parameters ({len(frozen)} components):")
    if len(frozen) > 0:
        print(f"  Total frozen: {sum(p.numel() for _, p in frozen):,}")
        print(f"  (Most of BERT backbone should be here)")
    
    if total_trainable < 10000:
        print("\n🔴 WARNING: Very few trainable params. Check LoRA injection.")
    
    return total_trainable, frozen

def check_lora_init(model):
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: LoRA MATRIX INITIALIZATION")
    print("="*70)
    
    lora_A_values = []
    lora_B_values = []
    
    for name, p in model.named_parameters():
        if 'A' in name and p.requires_grad:
            lora_A_values.append((name, p.data.clone()))
        elif 'B' in name and p.requires_grad:
            lora_B_values.append((name, p.data.clone()))
    
    print(f"\nFound {len(lora_A_values)} LoRA-A matrices and {len(lora_B_values)} LoRA-B matrices")
    
    if len(lora_A_values) > 0:
        A_sample = lora_A_values[0][1]
        print(f"\n✓ LoRA-A sample ({lora_A_values[0][0]}):")
        print(f"  Shape: {A_sample.shape}")
        print(f"  Mean: {A_sample.mean().item():.6f}")
        print(f"  Std:  {A_sample.std().item():.6f}")
        print(f"  Min:  {A_sample.min().item():.6f}")
        print(f"  Max:  {A_sample.max().item():.6f}")
        
        if (A_sample == 0).all():
            print("  🔴 CRITICAL: All zeros! LoRA won't contribute.")
        elif A_sample.std().item() < 1e-5:
            print("  🔴 WARNING: Near-zero std. Vanishing gradients likely.")
    
    return lora_A_values, lora_B_values

def test_forward_pass(model, batch_size=2, seq_length=128, device='cuda'):
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: FORWARD PASS")
    print("="*70)
    
    # dummy input
    input_ids = torch.randint(0, 30, (batch_size, seq_length), device=device)
    attention_mask = torch.ones(batch_size, seq_length, device=device)
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        try:
            # Sequence-level output
            emb = model(input_ids, attention_mask, per_residue=False)
            print(f"\n✓ Sequence-level embedding: {emb.shape}")
            print(f"  Mean: {emb.mean().item():.4f}, Std: {emb.std().item():.4f}")
            print(f"  Min: {emb.min().item():.4f}, Max: {emb.max().item():.4f}")
            
            if torch.isnan(emb).any():
                print("  🔴 WARNING: NaN in embeddings!")
            if torch.isinf(emb).any():
                print("  🔴 WARNING: Inf in embeddings!")
            
            # Per-residue output
            emb_per_res = model(input_ids, attention_mask, per_residue=True)
            print(f"\n✓ Per-residue embedding: {emb_per_res.shape}")
            print(f"  Mean: {emb_per_res.mean().item():.4f}, Std: {emb_per_res.std().item():.4f}")
            
            return emb, emb_per_res
        except Exception as e:
            print(f"\n🔴 ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def test_gradient_flow(model, task_heads, loss_fns, batch_size=2, device='cuda'):
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: GRADIENT FLOW")
    print("="*70)
    
    model.train()
    for head in task_heads:
        head.train()
    
    # dummy input and target
    input_ids = torch.randint(0, 30, (batch_size, 128), device=device)
    attention_mask = torch.ones(batch_size, 128, device=device)
    
    # Test task 0 (regression)
    print("\nTesting Task 0 (Regression - Thermostability):")
    targets = torch.randn(batch_size, 1, device=device)
    
    model = model.to(device)
    embeddings = model(input_ids, attention_mask, per_residue=False)
    logits = task_heads[0](embeddings)
    loss = loss_fns[0](logits.squeeze(-1), targets.squeeze(-1))
    
    print(f"  Loss: {loss.item():.6f}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_grad = False
    grad_norms = []
    
    for name, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:
            grad_norm = p.grad.norm().item()
            grad_norms.append((name, grad_norm))
            if grad_norm > 0:
                has_grad = True
    
    if has_grad:
        print(f"  ✓ Gradients flowing! Found {len(grad_norms)} parameters with gradients")
        grad_norms.sort(key=lambda x: x[1], reverse=True)
        print("  Top 5 gradient norms:")
        for name, norm in grad_norms[:5]:
            if norm > 1.0:
                print(f"    ⚠️  {name}: {norm:.6f} (very large!)")
            elif norm < 1e-6:
                print(f"    ⚠️  {name}: {norm:.6e} (very small!)")
            else:
                print(f"    ✓ {name}: {norm:.6f}")
    else:
        print(f"  🔴 CRITICAL: No gradients flowing!")
    
    return has_grad, grad_norms

def test_single_batch_overfit(model, task_heads, loss_fns, batch_size=4, num_iters=5, device='cuda'):
    print("\n" + "="*70)
    print("DIAGNOSTIC 5: SINGLE BATCH OVERFIT TEST")
    print("="*70)
    print("(If model works, loss should go to near-zero in 5 iterations)")
    
    model.train()
    for head in task_heads:
        head.train()
    
    model = model.to(device)
    
    # single batch
    input_ids = torch.randint(0, 30, (batch_size, 128), device=device)
    attention_mask = torch.ones(batch_size, 128, device=device)
    targets = torch.randn(batch_size, 1, device=device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(task_heads[0].parameters()),
        lr=1e-3
    )
    
    losses = []
    for i in range(num_iters):
        optimizer.zero_grad()
        
        embeddings = model(input_ids, attention_mask, per_residue=False)
        logits = task_heads[0](embeddings)
        loss = loss_fns[0](logits.squeeze(-1), targets.squeeze(-1))
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Iteration {i+1}: loss = {loss.item():.6f}")
    
    # Check if loss decreased
    improvement = losses[0] - losses[-1]
    pct_improvement = (improvement / losses[0]) * 100
    
    print(f"\n  Loss improvement: {improvement:.6f} ({pct_improvement:.1f}%)")
    
    if improvement > 0.1:
        print(f"  ✓ Model is learning! (Good sign)")
        return True
    else:
        print(f"  🔴 Model not learning on single batch. Something is frozen or broken.")
        return False


def run_full_diagnostic(device='cuda'):
    print("\n" + "#"*70)
    print("# MULTI-TASK PROTEIN MODEL DIAGNOSTIC SUITE")
    print("#"*70)
    
    # Load model
    print("\nLoading model...")
    model = SharedProtBert(lora=True, verbose=False)
    model = model.to(device)
    
    # Task heads
    from protbert_hf import build_regression_head, build_token_classification_head, build_sequence_classification_head
    hidden_dim = model.hidden_size
    task_heads = nn.ModuleList([
        build_regression_head(hidden_dim, 1),
        build_token_classification_head(hidden_dim, 8),
        build_sequence_classification_head(hidden_dim, 2)
    ])
    task_heads = task_heads.to(device)
    
    loss_fns = [
        nn.MSELoss(),
        nn.CrossEntropyLoss(ignore_index=-100),
        nn.CrossEntropyLoss(ignore_index=-100)
    ]
    
    # Run diagnostics
    diagnose_trainable_params(model)
    check_lora_init(model)
    test_forward_pass(model, device=device)
    has_grad, _ = test_gradient_flow(model, task_heads, loss_fns, device=device)
    
    if has_grad:
        can_overfit = test_single_batch_overfit(model, task_heads, loss_fns, device=device)
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    if not has_grad:
        print("🔴 STOP: Gradients not flowing. Check LoRA initialization.")
    elif can_overfit:
        print("✓ Model can learn. Issue is likely in:")
        print("  - Dataset quality or sampling")
        print("  - Hyperparameters (learning rate, batch size)")
        print("  - Task balancing (dynamic weights)")
    else:
        print("⚠️  Model structure OK but not learning. Check:")
        print("  - Gradient clipping is too aggressive")
        print("  - Learning rate is too low")
        print("  - Input data preprocessing")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_full_diagnostic(device=device)