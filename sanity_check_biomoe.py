import torch
import torch.nn as nn
from protbert_hf import SharedProtBert
from engine_hf_bio_moe import BioMoE_Engine

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_sanity_check():
    print("Starting Bio-MoE Sanity Check...")
    
    print("  [1] Initializing Backbone...")
    backbone = SharedProtBert(lora_rank=4, unfrozen_layers=0)
    
    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},       # Index 0
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8}, # Index 1
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}    # Index 2
    ]
    
    print("  [2] Initializing Bio-MoE Engine...")
    engine = BioMoE_Engine(backbone, task_configs, [], [], device=DEVICE)
    engine.train()
    
    # Create Dummy Data
    dummy_seqs = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", 
                  "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    
    inputs = backbone.tokenizer(dummy_seqs, return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
    targets = torch.randn(2, 1).to(DEVICE) # Dummy Regression Targets
    
    print("  [3] Running Forward Pass (with Bio-Routing)...")
    
    # --- UPDATE IS HERE: Unpack the tuple (logits, weights) ---
    logits, weights = engine.forward(inputs['input_ids'], inputs['attention_mask'], dummy_seqs, task_idx=0, debug=True)
    
    print(f"      -> Output Logits Shape: {logits.shape}")
    print(f"      -> Router Weights Shape: {weights.shape}")

    print("  [4] Computing Loss...")
    loss = nn.MSELoss()(logits, targets)
    
    print("  [5] Backpropagating...")
    loss.backward()
    
    print("\n  [6] 🕵️ Checking Gradients:")
    
    # 1. Did the Router learn?
    router_grad = False
    for p in engine.bio_router.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            router_grad = True
            break
            
    # 2. Did the Experts learn?
    expert_grad = False
    for p in engine.experts[0].parameters(): # Expert 0
        if p.grad is not None and p.grad.abs().sum() > 0:
            expert_grad = True
            break
            
    if router_grad and expert_grad:
        print("\n✅ SUCCESS: The Bio-Router and Experts are updating!")
        print("   The model is successfully using biophysics to route traffic.")
    else:
        print("\n❌ FAILURE: Gradients missing.")
        print(f"   Router Grad: {router_grad}")
        print(f"   Expert Grad: {expert_grad}")

if __name__ == "__main__":
    run_sanity_check()