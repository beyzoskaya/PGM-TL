import torch
import os
import shutil
import pandas as pd

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid

def run_hybrid_sanity_check_v2():
    print("\n🧬 STARTING SANITY CHECK V2: HYBRID (PCGrad + Uncertainty)...")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_DIR = "/content/drive/MyDrive/protein_multitask_outputs/test_sanity_check_hybrid_v2"
    if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)
    
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

    try:
        backbone = SharedProtBert(lora_rank=2, unfrozen_layers=0) 
        task_configs = [
            {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
            {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
            {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
        ]
        engine = MultiTaskEngineHybrid(
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
  
        initial_sigma = engine.log_vars.clone().detach()
        
        trainable_param = None
        for p in engine.parameters():
            if p.requires_grad:
                trainable_param = p
                break
        
        if trainable_param is None:
            print("❌ Critical Error: No trainable parameters found in model!")
            return
            
        initial_weight_snap = trainable_param.clone().detach()
        print(f"   Snapshotting trainable param (Size: {initial_weight_snap.shape})")

    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    print("\n[Step 3] Running Training Steps...")
    engine.train()
    
    try:
        # Run for a few steps
        engine.train_one_epoch(optimizer, epoch_index=1)
        print("   ✅ train_one_epoch finished without crashing.")

    except Exception as e:
        print(f"\n❌ CRITICAL FAIL: {e}")
        return

    print("\n[Step 4] Verifying Updates...")
    
    # Check Sigmas
    final_sigma = engine.log_vars.detach()
    if not torch.equal(initial_sigma, final_sigma):
        print(f"   ✅ Sigmas Updated: {initial_sigma.cpu().numpy()} -> {final_sigma.cpu().numpy()}")
    else:
        print("   ⚠️ WARNING: Sigmas did not change!")

    # Check Weights
    final_weight_snap = trainable_param.detach()
    if not torch.equal(initial_weight_snap, final_weight_snap):
        diff = (final_weight_snap - initial_weight_snap).abs().sum().item()
        print(f"   ✅ Weights Updated (Diff: {diff:.6f}). PCGrad worked.")
    else:
        print("   ❌ CRITICAL: Trainable weights did not change!")

    print("\n🚀 SANITY CHECK COMPLETE.")

if __name__ == "__main__":
    run_hybrid_sanity_check_v2()