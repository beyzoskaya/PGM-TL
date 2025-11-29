import torch
import torch.nn as nn
from torch.utils.data import Subset

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_prompt import TaskPromptedEngine

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_real_data_sanity_check():
    print("Starting Prompt-Tuning Sanity Check (Real Data Mode)...")

    print("  [1] Loading Data Subsets...")
    ds_t = Thermostability(verbose=0)
    ds_s = SecondaryStructure(verbose=0)
    ds_c = CloningCLF(verbose=0)
    
    # Create mini-datasets (4 samples each)
    train_t = Subset(ds_t, range(4))
    train_s = Subset(ds_s, range(4))
    train_c = Subset(ds_c, range(4))
    
    print(f"      -> Loaded {len(train_t)} Thermo, {len(train_s)} SSP, {len(train_c)} Cloning samples.")

    print("  [2] Initializing Model & Engine...")
    # Match training config: Unfrozen layers required for prompt interaction
    backbone = SharedProtBert(lora_rank=4, unfrozen_layers=2) 
    
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},       # Index 0
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8}, # Index 1
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}    # Index 2
    ]
    
    # Initialize Engine with REAL (but small) datasets
    engine = TaskPromptedEngine(
        backbone=backbone, 
        task_configs=task_configs, 
        train_sets=[train_t, train_s, train_c], # Real loaders will be created here
        valid_sets=None, 
        test_sets=None,
        batch_size=2, 
        device=DEVICE
    )
    engine.train()
    
    print("  [3] Fetching a Real Batch for Task 0 (Thermostability)...")
    # Manually grab the first batch from the Thermo loader
    loader_iter = iter(engine.train_loaders[0])
    batch = next(loader_iter)
    
    input_ids = batch['input_ids'].to(DEVICE)
    mask = batch['attention_mask'].to(DEVICE)
    targets = batch['targets'].to(DEVICE)
    
    print(f"      -> Input Shape: {input_ids.shape}")
    print(f"      -> Target Shape: {targets.shape}")

    # 4. FORWARD PASS
    print("  [4] Running Forward Pass with Task Prompt 0...")
    # Explicitly pass task_idx=0 to activate Thermo Prompt
    logits = engine.forward(input_ids, mask, task_idx=0, debug=True)
    
    # 5. BACKWARD PASS
    print("  [5] Computing Loss & Backprop...")
    loss = engine.loss_fns[0](logits, targets)
    loss.backward()
    
    # 6. GRADIENT VERIFICATION
    print("\n  [6] 🕵️ Checking Gradients:")
    
    # A. Check Prompt 0 (Thermo) -> Should Update
    prompt_0_grad = False
    if engine.task_prompts[0].grad is not None:
        grad_sum = engine.task_prompts[0].grad.abs().sum().item()
        if grad_sum > 0:
            prompt_0_grad = True
            print(f"      ✅ Prompt 0 (Thermo) has gradients! (Sum: {grad_sum:.4f})")
        else:
             print(f"      ❌ Prompt 0 has Zero gradients.")
    else:
        print(f"      ❌ Prompt 0 has None gradients.")

    # B. Check Prompt 2 (Cloning) -> Should Stay Frozen (Isolation Check)
    prompt_2_grad = True
    if engine.task_prompts[2].grad is None or engine.task_prompts[2].grad.abs().sum().item() == 0:
        prompt_2_grad = False # This is good!
        print("      ✅ Prompt 2 (Cloning) has NO gradients (Correct isolation).")
    else:
        print("      ❌ Prompt 2 has gradients! (Task Leakage detected)")

    # C. Check Backbone -> Should Update (because unfrozen_layers=2)
    backbone_grad = False
    for n, p in engine.backbone.named_parameters():
        if p.grad is not None and p.requires_grad:
            backbone_grad = True
            # print(f"      (Debug) Gradient found in: {n}") # Uncomment if needed
            break
    
    if backbone_grad:
        print("      ✅ Backbone is receiving updates (Integration successful).")
    else:
        print("      ❌ Backbone has no gradients. (Did you set unfrozen_layers=0? It needs to be >0 for prompting).")

    if prompt_0_grad and not prompt_2_grad and backbone_grad:
        print("\nREAL DATA SANITY CHECK PASSED: The Framework is valid.")
    else:
        print("\nSANITY CHECK FAILED. Do not train.")

if __name__ == "__main__":
    run_real_data_sanity_check()