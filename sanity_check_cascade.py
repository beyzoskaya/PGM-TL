import torch
import torch.nn as nn
from protbert_hf import SharedProtBert
from engine_hf_cascade import CascadeMultiTaskEngine

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_gradient_check():
    print("🔬 Starting Cascade Gradient Sanity Check...")

    print("  [1] Initializing Model...")
    backbone = SharedProtBert(lora_rank=4, unfrozen_layers=0) # Small for speed
    
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    engine = CascadeMultiTaskEngine(backbone, task_configs, [], [], device=DEVICE)
    engine.train()

    input_ids = torch.randint(0, 100, (2, 10)).to(DEVICE)
    attention_mask = torch.ones((2, 10)).to(DEVICE)
    dummy_thermo_target = torch.randn(2, 1).to(DEVICE) # Regression target
    
    print("  [2] Running Forward Pass...")
    ssp_logits, thermo_pred, clone_pred = engine.forward(input_ids, attention_mask, debug=True)
    
    print("  [3] Computing Loss on Thermostability ONLY...")
    loss = nn.MSELoss()(thermo_pred, dummy_thermo_target)
    
    print("  [4] Backpropagating...")
    loss.backward()
    
    # 6. CHECK GRADIENTS
    # The Critical Test: Does the Structure Head have gradients?
    # If YES: Thermo task is updating Structure weights (Cascade works!)
    # If NO:  Thermo task is independent (Cascade failed)
    
    print("\n  [5] 🕵️ Checking Gradients:")

    adapter_grad = False
    for p in engine.cascade_adapter.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            adapter_grad = True
            break
            
    ssp_head_grad = False
    for p in engine.head_ssp.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            ssp_head_grad = True
            break
            
    if adapter_grad and ssp_head_grad:
        print("\n✅ SUCCESS: Gradients flowed from Thermostability -> Adapter -> Structure Head.")
        print("   This confirms the hierarchical connection is active.")
    else:
        print("\n❌ FAILURE: No gradients found in Structure path.")
        print(f"   Adapter Grad: {adapter_grad}")
        print(f"   Structure Head Grad: {ssp_head_grad}")

if __name__ == "__main__":
    run_gradient_check()