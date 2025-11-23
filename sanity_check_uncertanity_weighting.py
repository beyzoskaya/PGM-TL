import torch
import torch.nn as nn
import os
import shutil
import numpy as np
from torch.utils.data import DataLoader

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_with_uncertanity_weighting import MultiTaskEngineUncertanityWeighting

def run_robust_sanity_check():
    print("\nSTARTING ROBUST SANITY CHECK (UNCERTAINTY WEIGHTING)...")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_DIR = "/content/drive/MyDrive/protein_multitask_outputs/test_sanity_check"
    if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)
    
    print(f"[Setup] Running on {DEVICE}. Temp dir: {TEST_DIR}")

    print("[Step 1] Loading Mock Data...")
    try:
        ds_thermo = Thermostability(verbose=0)
        ds_ssp = SecondaryStructure(verbose=0)
        ds_clf = CloningCLF(verbose=0)
        
        train_sets = [torch.utils.data.Subset(ds_thermo, range(4)), 
                      torch.utils.data.Subset(ds_ssp, range(4)), 
                      torch.utils.data.Subset(ds_clf, range(4))]
    except Exception as e:
        print(f"❌ Data Load Failed: {e}")
        return

    print("[Step 2] Initializing Model & Engine...")
    try:
        backbone = SharedProtBert(lora_rank=2, unfrozen_layers=0) 
        
        task_configs = [
            {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
            {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
            {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
        ]
        
        engine = MultiTaskEngineUncertanityWeighting(
            backbone=backbone,
            task_configs=task_configs,
            train_sets=train_sets,
            valid_sets=train_sets,
            batch_size=2,
            device=DEVICE,
            save_dir=TEST_DIR
        )
        optimizer = torch.optim.AdamW(engine.parameters(), lr=1e-3)
        print("   ✅ Engine Initialized.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return
    print("\n[Step 3] Testing Critical 'Retain Graph' Fix...")
    
    engine.train()
    engine.backbone.train()
    
    iterators = [iter(l) for l in engine.train_loaders]
    batches = [next(it) for it in iterators]
    
    optimizer.zero_grad()
    raw_losses_list = []
    
    try:
        for i in range(3):
            batch = batches[i]
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            is_token = (task_configs[i]['type'] == 'token_classification')
            
            embeddings = engine.backbone(ids, mask, task_type='token' if is_token else 'sequence')
            logits = engine.heads[i](embeddings)
            
            if is_token:
                loss = engine.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
            else:
                loss = engine.loss_fns[i](logits, targets)
            
            raw_losses_list.append(loss)

            # Uncertainty Logic
            precision = torch.exp(-engine.log_vars[i])
            weighted_loss = (precision * loss) + engine.log_vars[i]
            
            print(f"   > Backward Task {i} (retain_graph=True)...")
            weighted_loss.backward(retain_graph=True) 
        
        print("   ✅ Backward pass with retain_graph=True successful.")
        
        print("   > Analyzing Gradients (Accessing retained graph)...")
        engine.analyze_gradients(raw_losses_list, step=0, epoch=1)
        print("   ✅ Gradient Analysis successful (Fix Verified!).")
        
        optimizer.step()
        print("   ✅ Optimizer step successful.")
        
    except RuntimeError as e:
        print(f"\n❌ CRITICAL FAIL: {e}")
        print("HINT: Did you update 'engine_hf_with_uncertanity_weighting.py' with the new code?")
        return
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        return

    print("\n[Step 4] Checking Sigma Updates...")
    print(f"   Current Log Vars: {engine.log_vars.detach().cpu().numpy()}")
    if torch.all(engine.log_vars == 0.0):
        print("   ⚠️ WARNING: Sigmas haven't moved from 0.0 yet (might need more steps, but check grads).")
        if engine.log_vars.grad is not None:
             print(f"      But Gradients exist! Max grad: {engine.log_vars.grad.abs().max()}")
             print("      ✅ Gradients are flowing.")
        else:
             print("      ❌ No gradients on log_vars!")
    else:
        print("   ✅ Sigmas updated (Values changed).")

    print("\n[Step 5] Checking Logs...")
    if os.path.exists(os.path.join(TEST_DIR, "gradient_conflict_log.txt")):
        print("   ✅ gradient_conflict_log.txt created.")
    else:
        print("   ❌ gradient_conflict_log.txt MISSING.")

    print("\nSANITY CHECK COMPLETE. You are ready to run main_hf_v2.py")

if __name__ == "__main__":
    run_robust_sanity_check()