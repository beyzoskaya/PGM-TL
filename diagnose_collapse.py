import torch
import torch.nn as nn
from protbert_hf import SharedProtBert, build_regression_head
import numpy as np

def test_gradient_explosion_detailed(device='cuda'):
   
    print("\n" + "="*70)
    print("GRADIENT EXPLOSION DIAGNOSTIC")
    print("="*70)
    
    model = SharedProtBert(lora=True, verbose=False)
    model = model.to(device)
    
    hidden_dim = model.hidden_size
    task_head = build_regression_head(hidden_dim, 1)
    task_head = task_head.to(device)
    
    loss_fn = nn.MSELoss()
    
    # dummy batch
    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, 30, (batch_size, seq_length), device=device)
    attention_mask = torch.ones(batch_size, seq_length, device=device)
    targets = torch.randn(batch_size, 1, device=device)
    
    model.train()
    task_head.train()
    
    print("\n[TEST 1] Gradient magnitudes before clipping")
    print("-" * 70)
    
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask, per_residue=False)
    
    logits = task_head(embeddings)
    loss = loss_fn(logits.squeeze(-1), targets.squeeze(-1))
    
    print(f"Loss: {loss.item():.6f}")
    
    # Backward WITHOUT clipping
    loss.backward()
    
    # Check LoRA-specific gradients
    lora_a_grads = []
    lora_b_grads = []
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            if 'A' in name and 'layer.0' in name:
                lora_a_grads.append((name, grad_norm))
            elif 'B' in name and 'layer.0' in name:
                lora_b_grads.append((name, grad_norm))
    
    print("\nLayer 0 LoRA-A gradient norms:")
    for name, norm in lora_a_grads:
        print(f"  {name}: {norm:.6e}")
    
    print("\nLayer 0 LoRA-B gradient norms (SUSPECT - often explodes):")
    for name, norm in lora_b_grads:
        status = "✓ OK" if norm < 1.0 else "🔴 VERY LARGE" if norm > 10.0 else "⚠️  Large"
        print(f"  {name}: {norm:.6e} {status}")
    
    # Check all gradients
    all_grads = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            all_grads.append(p.grad.norm().item())
    
    print(f"\nGradient statistics (all parameters):")
    print(f"  Mean: {np.mean(all_grads):.6e}")
    print(f"  Std:  {np.std(all_grads):.6e}")
    print(f"  Min:  {np.min(all_grads):.6e}")
    print(f"  Max:  {np.max(all_grads):.6e}")
    print(f"  Num grads > 1.0: {sum(1 for g in all_grads if g > 1.0)} / {len(all_grads)}")
    
    # Clear gradients
    model.zero_grad()
    task_head.zero_grad()
    
    print("\n[TEST 2] Effect of gradient clipping")
    print("-" * 70)
    
    # Forward again
    embeddings = model(input_ids, attention_mask, per_residue=False)
    logits = task_head(embeddings)
    loss = loss_fn(logits.squeeze(-1), targets.squeeze(-1))
    loss.backward()
    
    # Get unclipped norm
    all_params = list(model.parameters()) + list(task_head.parameters())
    
    total_norm_before = 0
    for p in all_params:
        if p.grad is not None:
            total_norm_before += p.grad.norm().item() ** 2
    total_norm_before = float(np.sqrt(total_norm_before))
    
    print(f"Total gradient norm BEFORE clipping: {total_norm_before:.6e}")
    
    # Apply different clipping thresholds
    for clip_value in [0.5, 1.0, 5.0, 10.0]:
        # Make a copy of gradients
        grads_before = [p.grad.clone() if p.grad is not None else None for p in all_params]
        
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=clip_value)
        
        total_norm_after = 0
        for p in all_params:
            if p.grad is not None:
                total_norm_after += p.grad.norm().item() ** 2
        total_norm_after = float(np.sqrt(total_norm_after))
        
        # Restore
        for p, grad in zip(all_params, grads_before):
            if grad is not None:
                p.grad = grad.clone()
        
        status = "✓ OK" if total_norm_after < 1.0 else "⚠️  Still large"
        print(f"  Clip at {clip_value:4.1f}: {total_norm_after:.6e} {status}")
    
    print("\n[TEST 3] Checking LoRA B matrix initialization impact")
    print("-" * 70)
    
    print("Current LoRA-B initialization strategy: zeros")
    print("This can lead to unstable gradients when combined with non-zero A")
    print("\nProposed fix: Initialize B with small random values too")
    
    # Check if B matrices are actually learning
    model.zero_grad()
    task_head.zero_grad()
    
    embeddings = model(input_ids, attention_mask, per_residue=False)
    logits = task_head(embeddings)
    loss = loss_fn(logits.squeeze(-1), targets.squeeze(-1))
    loss.backward()
    
    print("\nLoRA-B matrix gradients (should NOT be zero):")
    b_grads_found = 0
    for name, p in model.named_parameters():
        if 'B' in name and p.grad is not None and p.grad.norm().item() > 0:
            b_grads_found += 1
    
    print(f"  Non-zero B gradients: {b_grads_found} (good, they're learning)")
    
    print("\n[TEST 4] Check for NaN/Inf in gradients")
    print("-" * 70)
    
    has_nan = False
    has_inf = False
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                print(f"  🔴 NaN found in {name}")
                has_nan = True
            if torch.isinf(p.grad).any():
                print(f"  🔴 Inf found in {name}")
                has_inf = True
    
    for name, p in task_head.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                print(f"  🔴 NaN found in {name}")
                has_nan = True
            if torch.isinf(p.grad).any():
                print(f"  🔴 Inf found in {name}")
                has_inf = True
    
    if not has_nan and not has_inf:
        print("  ✓ No NaN or Inf gradients")
    
    print("\n[TEST 5] Step-by-step gradient tracking through optimization")
    print("-" * 70)
    
    # Reset
    model.zero_grad()
    task_head.zero_grad()
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(task_head.parameters()),
        lr=5e-4
    )
    
    print("Tracking 3 optimization steps:")
    losses = []
    grad_norms = []
    
    for step in range(3):
        optimizer.zero_grad()
        
        embeddings = model(input_ids, attention_mask, per_residue=False)
        logits = task_head(embeddings)
        loss = loss_fn(logits.squeeze(-1), targets.squeeze(-1))
        
        loss.backward()
        
        # Compute grad norm before clipping
        total_norm = 0
        for p in list(model.parameters()) + list(task_head.parameters()):
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = float(np.sqrt(total_norm))
        
        # Clip
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(task_head.parameters()),
            max_norm=1.0
        )
        
        # Step
        optimizer.step()
        
        losses.append(loss.item())
        grad_norms.append(total_norm)
        
        status = "✓ Decreasing" if len(losses) < 2 or losses[-1] < losses[-2] else "🔴 Increasing"
        print(f"  Step {step+1}: loss={loss.item():.6f} {status} | grad_norm_pre_clip={total_norm:.6e}")
    
    # Check trend
    print("\nLoss trend:")
    improvement = losses[0] - losses[-1]
    if improvement > 0.1:
        print(f"  ✓ Loss improved by {improvement:.4f} (good)")
    else:
        print(f"  🔴 Loss did NOT improve: {improvement:.4f}")


def test_lora_b_init_fix(device='cuda'):
    print("\n" + "="*70)
    print("LORA-B INITIALIZATION FIX TEST")
    print("="*70)
    
    from protbert_hf import LoRALinear
    
    print("\nTesting different LoRA-B initialization strategies:")
    
    for init_strategy in ['zeros', 'normal_small', 'normal_large']:
        print(f"\n[Strategy: {init_strategy}]")
        
        # Create LoRA layer
        lora = LoRALinear(1024, 1024, r=16, alpha=32, dropout=0.1)
        lora = lora.to(device)
        
        # Re-initialize B based on strategy
        if init_strategy == 'zeros':
            nn.init.zeros_(lora.B)
        elif init_strategy == 'normal_small':
            nn.init.normal_(lora.B, std=0.02)
        else:  # normal_large
            nn.init.normal_(lora.B, std=0.1)
        
        # Test on dummy data
        x = torch.randn(4, 1024, device=device, requires_grad=True)
        output = lora(x)
        loss = output.mean()
        loss.backward()
        
        # Check gradient variance
        b_grad_norm = lora.B.grad.norm().item()
        
        print(f"  LoRA-B gradient norm: {b_grad_norm:.6e}")
        print(f"  LoRA-B initial values: min={lora.B.min().item():.6f}, max={lora.B.max().item():.6f}")
        
        if b_grad_norm > 1.0:
            print(f"  ⚠️  Gradient norm is large - may cause instability")
        else:
            print(f"  ✓ Gradient norm is reasonable")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    test_gradient_explosion_detailed(device=device)
    test_lora_b_init_fix(device=device)
    
    print("\n" + "="*70)
    print("RECOMMENDED FIXES (in order of likelihood):")
    print("="*70)
    print("""
1. 🔴 PRIMARY: Initialize LoRA-B with small random values, NOT zeros
   - Change: nn.init.zeros_(self.B) → nn.init.normal_(self.B, std=0.02)
   - Reason: Zero B + non-zero A leads to gradient concentration in B during early training

2. 🟡 SECONDARY: Increase gradient clipping threshold
   - Current: 1.0 might be too aggressive
   - Try: 5.0 or 10.0 first, then reduce gradually

3. 🟢 TERTIARY: Reduce learning rate slightly
   - Current: 5e-4 is reasonable but might be high during instability
   - Try: 2e-4 or 3e-4 with better initialization

4. 🟢 BONUS: Add layer-wise learning rate decay
   - Helps with LoRA training stability
    """)