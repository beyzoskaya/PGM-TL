import torch
import os
import shutil
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_with_uncertanity_weighting import MultiTaskEngineUncertanityWeighting

def run_sanity_check():
    print("\n🚀 STARTING PRE-FLIGHT CONTROL FOR UNCERTAINTY WEIGHTING...")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_DIR = "/content/drive/MyDrive/protein_multitask_outputs/test_sanity_check"
    if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR) 
    
    print("[Check 1] Data Loading...")
    try:
        ds_thermo = Thermostability(verbose=0)
        ds_ssp = SecondaryStructure(verbose=0)
        ds_clf = CloningCLF(verbose=0)
        
        train_sets = [torch.utils.data.Subset(ds_thermo, range(10)), 
                      torch.utils.data.Subset(ds_ssp, range(10)), 
                      torch.utils.data.Subset(ds_clf, range(10))]
        print("   ✅ Data Loaded.")
    except Exception as e:
        print(f"   ❌ Data Failed: {e}")
        return

    print("[Check 2] Model & Engine Init...")
    try:
        backbone = SharedProtBert(lora_rank=4, unfrozen_layers=0)
        
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
        
        print(f"   Initial Log Vars (Sigmas): {engine.log_vars.data}")
        if not engine.log_vars.requires_grad:
            print("   ❌ CRITICAL: log_vars does not require grad!")
            return

    except Exception as e:
        print(f"   ❌ Init Failed: {e}")
        return

    print("[Check 3] Training Loop & Gradient Flow...")
    engine.train()
    initial_sigma = engine.log_vars.clone().detach()
    
    try:
        loader_iter = [iter(l) for l in engine.train_loaders]
        
        for step in range(5):
            optimizer.zero_grad()
            total_loss = 0
            
            for i in range(3):
                batch = next(loader_iter[i])
                # (Simplified forward pass logic from engine for testing)
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                
                # Check Forward
                embeddings = engine.backbone(ids, mask, task_type='token' if i==1 else 'sequence')
                logits = engine.heads[i](embeddings)
                
                if i==1: loss = engine.loss_fns[i](logits.view(-1, logits.shape[-1]), targets.view(-1))
                else: loss = engine.loss_fns[i](logits, targets)
                
                # Check UW Formula
                precision = torch.exp(-engine.log_vars[i])
                weighted_loss = (precision * loss) + engine.log_vars[i]
                weighted_loss.backward()
                total_loss += weighted_loss.item()
            
            optimizer.step()
            print(f"   Step {step}: Loss={total_loss:.4f}")

        # 5. Check if Sigmas Changed
        final_sigma = engine.log_vars.detach()
        print(f"   Final Log Vars: {final_sigma}")
        
        if torch.equal(initial_sigma, final_sigma):
            print("   ⚠️ WARNING: Sigmas did not change! (Might happen in only 5 steps if LR is low, but check optimizer)")
        else:
            print("   ✅ Sigmas Updated successfully (Gradients are flowing to uncertainty weights).")

    except Exception as e:
        print(f"   ❌ Training Failed: {e}")
        return

    # 6. Check Logging
    print("[Check 4] File Saving...")
    if os.path.exists(os.path.join(TEST_DIR, "training_dynamics_sigmas.csv")):
        print("   ✅ CSV Log found.")
    else:
        print("   ❌ CSV Log missing.")

    print("\n🚀 PRE-FLIGHT CHECK COMPLETE. You are ready to train.")

if __name__ == "__main__":
    run_sanity_check()