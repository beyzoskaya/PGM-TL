import torch
import torch.nn as nn
from protbert_hf import SharedProtBert, build_regression_head
import numpy as np

def test_gradient_explosion_detailed(device='cuda'):
    """
    Detailed test to check:
    1. If gradients explode during backward
    2. Which layers have the largest gradients
    3. If LoRA B matrix is causing issues
    4. Effect of gradient clipping
    """
    print("\n" + "="*70)
    print("GRADIENT EXPLOSION DIAGNOSTIC")
    print("="*70)
    
    # Setup
    print("\nLoading model...")
    model = SharedProtBert(lora=True, verbose=False)
    model = model.to(device)
    
    hidden_dim = model.hidden_size
    task_head = build_regression_head(hidden_dim, 1)
    task_head = task_head.to(device)
    
    loss_fn = nn.MSELoss()
    
    # Create dummy batch
    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, 30, (batch_size, seq_length), device=device)
    attention_mask = torch.ones(batch_size, seq_length, device=device)
    targets = torch.randn(batch_size, 1, device=device)
    
    model.train()
    task_head.train()
    
    print("\n[TEST 1] Forward pass and backward")
    print("-" * 70)
    
    embeddings = model(input_ids, attention_mask, per_residue=False)
    logits = task_head(embeddings)
    loss = loss_fn(logits.squeeze(-1), targets.squeeze(-1))
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Backward
    loss.backward()
    
    print("\n[TEST 2] Collecting all gradients")
    print("-" * 70)
    
    all_grads = []
    lora_a_grads = []
    lora_b_grads = []
    task_head_grads = []
    
    # Collect from backbone
    for name, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:
            grad_norm = p.grad.norm().item()
            all_grads.append(grad_norm)
            
            if 'A' in name:
                lora_a_grads.append((name, grad_norm))
            elif 'B' in name:
                lora_b_grads.append((name, grad_norm))
    
    # Collect from task head
    for name, p in task_head.named_parameters():
        if p.grad is not None and p.requires_grad:
            grad_norm = p.grad.norm().item()
            all_grads.append(grad_norm)
            task_head_grads.append((name, grad_norm))
    
    print(f"Total gradients collected: {len(all_grads)}")
    print(f"  LoRA-A gradients: {len(lora_a_grads)}")
    print(f"  LoRA-B gradients: {len(lora_b_grads)}")
    print(f"  Task head gradients: {len(task_head_grads)}")
    
    if len(all_grads) == 0:
        print("\n🔴 CRITICAL: No gradients found! Something is wrong with backward pass.")
        return
    
    print("\n[TEST 3] LoRA-A gradient statistics")
    print("-" * 70)
    if len(lora_a_grads) > 0:
        a_norms = [g for _, g in lora_a_grads]
        print(f"Found {len(lora_a_grads)} LoRA-A matrices")
        print(f"  Mean: {np.mean(a_norms):.6e}")
        print(f"  Std:  {np.std(a_norms):.6e}")
        print(f"  Min:  {np.min(a_norms):.6e}")
        print(f"  Max:  {np.max(a_norms):.6e}")
        print("\nTop 3 LoRA-A gradient norms:")
        for name, norm in sorted(lora_a_grads, key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {name}: {norm:.6e}")
    else:
        print("⚠️  No LoRA-A gradients found")
    
    print("\n[TEST 4] LoRA-B gradient statistics (SUSPECT - often explodes)")
    print("-" * 70)
    if len(lora_b_grads) > 0:
        b_norms = [g for _, g in lora_b_grads]
        print(f"Found {len(lora_b_grads)} LoRA-B matrices")
        print(f"  Mean: {np.mean(b_norms):.6e}")
        print(f"  Std:  {np.std(b_norms):.6e}")
        print(f"  Min:  {np.min(b_norms):.6e}")
        print(f"  Max:  {np.max(b_norms):.6e}")
        
        print("\nTop 3 LoRA-B gradient norms:")
        for name, norm in sorted(lora_b_grads, key=lambda x: x[1], reverse=True)[:3]:
            status = "✓ OK" if norm < 1.0 else "🔴 VERY LARGE" if norm > 10.0 else "⚠️  Large"
            print(f"  {name}: {norm:.6e} {status}")
    else:
        print("⚠️  No LoRA-B gradients found")
    
    print("\n[TEST 5] Task head gradient statistics")
    print("-" * 70)
    if len(task_head_grads) > 0:
        head_norms = [g for _, g in task_head_grads]
        print(f"Found {len(task_head_grads)} task head parameters with gradients")
        print(f"  Mean: {np.mean(head_norms):.6e}")
        print(f"  Std:  {np.std(head_norms):.6e}")
        print(f"  Min:  {np.min(head_norms):.6e}")
        print(f"  Max:  {np.max(head_norms):.6e}")
    else:
        print("⚠️  No task head gradients found")
    
    print("\n[TEST 6] Overall gradient statistics (all parameters)")
    print("-" * 70)
    print(f"  Mean: {np.mean(all_grads):.6e}")
    print(f"  Std:  {np.std(all_grads):.6e}")
    print(f"  Min:  {np.min(all_grads):.6e}")
    print(f"  Max:  {np.max(all_grads):.6e}")
    print(f"  Num grads > 1.0: {sum(1 for g in all_grads if g > 1.0)} / {len(all_grads)}")
    print(f"  Num grads > 10.0: {sum(1 for g in all_grads if g > 10.0)} / {len(all_grads)}")
    
    # Clear gradients
    model.zero_grad()
    task_head.zero_grad()
    
    print("\n[TEST 7] Effect of gradient clipping")
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
    
    # Apply clipping at different thresholds
    print("\nGradient norm at different clipping thresholds:")
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
        
        reduction_pct = ((total_norm_before - total_norm_after) / total_norm_before * 100) if total_norm_before > 0 else 0
        status = "✓ OK" if total_norm_after < 1.0 else "⚠️  Still large" if total_norm_after < 5.0 else "🔴 Very large"
        print(f"  Clip {clip_value:4.1f}: {total_norm_after:.6e} ({reduction_pct:5.1f}% reduction) {status}")
    
    print("\n[TEST 8] Multi-step optimization test")
    print("-" * 70)
    
    # Reset
    model.zero_grad()
    task_head.zero_grad()
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(task_head.parameters()),
        lr=2e-4
    )
    
    print("Tracking 5 optimization steps with lr=2e-4, clip=1.0:")
    losses = []
    grad_norms_before_clip = []
    grad_norms_after_clip = []
    
    for step in range(5):
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
        grad_norms_before_clip.append(total_norm)
        
        # Clip
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(task_head.parameters()),
            max_norm=1.0
        )
        
        # Compute grad norm after clipping
        total_norm_clipped = 0
        for p in list(model.parameters()) + list(task_head.parameters()):
            if p.grad is not None:
                total_norm_clipped += p.grad.norm().item() ** 2
        total_norm_clipped = float(np.sqrt(total_norm_clipped))
        grad_norms_after_clip.append(total_norm_clipped)
        
        # Step
        optimizer.step()
        
        losses.append(loss.item())
        
        status = "✓" if len(losses) < 2 or losses[-1] < losses[-2] else "🔴"
        print(f"  Step {step+1}: loss={loss.item():.6f} {status} | grad_norm={total_norm:.6e} → {total_norm_clipped:.6e}")
    
    # Check trend
    print("\nLoss trend analysis:")
    improvements = [losses[i] - losses[i+1] for i in range(len(losses)-1)]
    avg_improvement = np.mean(improvements)
    
    if avg_improvement > 0.05:
        print(f"  ✓ Average loss improvement per step: {avg_improvement:.6f} (GOOD)")
    elif avg_improvement > 0.01:
        print(f"  ⚠️  Average loss improvement per step: {avg_improvement:.6f} (SLOW)")
    else:
        print(f"  🔴 Average loss improvement per step: {avg_improvement:.6f} (NOT LEARNING)")
    
    # Stability analysis
    increasing_steps = sum(1 for i in range(len(losses)-1) if losses[i+1] > losses[i])
    if increasing_steps == 0:
        print(f"  ✓ Loss monotonically decreasing (STABLE)")
    elif increasing_steps <= 1:
        print(f"  ⚠️  Loss increased in {increasing_steps} step (mostly stable)")
    else:
        print(f"  🔴 Loss increased in {increasing_steps} steps (UNSTABLE)")


def check_lora_init_values(device='cuda'):
    """Check if LoRA matrices are properly initialized"""
    print("\n" + "="*70)
    print("LoRA MATRIX INITIALIZATION VALUES")
    print("="*70)
    
    model = SharedProtBert(lora=True, verbose=False)
    model = model.to(device)
    
    print("\nChecking LoRA-A and LoRA-B initialization values:")
    
    lora_a_vals = []
    lora_b_vals = []
    
    for name, p in model.named_parameters():
        if 'A' in name and p.requires_grad:
            lora_a_vals.append(p.data.clone())
        elif 'B' in name and p.requires_grad:
            lora_b_vals.append(p.data.clone())
    
    if len(lora_a_vals) > 0:
        a_concat = torch.cat([v.flatten() for v in lora_a_vals])
        print(f"\nLoRA-A matrices ({len(lora_a_vals)} total):")
        print(f"  Mean: {a_concat.mean().item():.6f}")
        print(f"  Std:  {a_concat.std().item():.6f}")
        print(f"  Min:  {a_concat.min().item():.6f}")
        print(f"  Max:  {a_concat.max().item():.6f}")
        if a_concat.std().item() > 0.01:
            print(f"  ✓ A matrices are properly initialized (non-zero)")
        else:
            print(f"  🔴 A matrices are too close to zero!")
    
    if len(lora_b_vals) > 0:
        b_concat = torch.cat([v.flatten() for v in lora_b_vals])
        print(f"\nLoRA-B matrices ({len(lora_b_vals)} total):")
        print(f"  Mean: {b_concat.mean().item():.6f}")
        print(f"  Std:  {b_concat.std().item():.6f}")
        print(f"  Min:  {b_concat.min().item():.6f}")
        print(f"  Max:  {b_concat.max().item():.6f}")
        if b_concat.std().item() > 0.01:
            print(f"  ✓ B matrices are properly initialized (non-zero)")
        elif (b_concat == 0).all().item():
            print(f"  🔴 B matrices are all ZEROS (should have been fixed!)")
        else:
            print(f"  ⚠️  B matrices are close to zero")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    check_lora_init_values(device=device)
    test_gradient_explosion_detailed(device=device)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
If LoRA-B matrices show Std: 0.000000 (all zeros):
  ✗ Your fixed protbert_hf.py is NOT being used
  → Make sure you saved the updated version correctly
  → Restart Python/Colab kernel to reload modules
  → Check file was updated: grep -n "nn.init.normal_(self.B" protbert_hf.py

If gradients are flowing but still large (> 1.0 frequently):
  → Consider increasing gradient clipping: max_norm=5.0
  → Or reduce learning rate: lr=1e-4

If loss is decreasing but slowly:
  → Training is working! Just needs more epochs
  → Consider increasing learning rate: lr=5e-4

If loss is chaotic (up and down):
  → Try reducing learning rate by half: 1e-4
  → Or increase gradient clipping: max_norm=5.0
    """)