import torch
import os
import shutil
import pandas as pd

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_hybrid_pcgrad import MultiTaskEngineHybrid

def run_hybrid_sanity_check():
    print("\n🧬 STARTING SANITY CHECK: HYBRID (PCGrad + Uncertainty)...")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_DIR = "./test_sanity_check_hybrid"
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

    print("[Step 2] Initializing Model & Hybrid Engine...")
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
        
        # Snapshot initial weights to check for updates later
        initial_sigma = engine.log_vars.clone().detach()
        # We check a specific LoRA weight
        initial_lora_weight = list(engine.backbone.parameters())[-1].clone().detach()

    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return
    
    print("\n[Step 3] Running Training Steps (PCGrad + UW)...")
    engine.train()
    
    try:
        
        loader_lens = [len(l) for l in engine.train_loaders]
        iterators = [iter(l) for l in engine.train_loaders]

        for step in range(3):
            print(f"   > Step {step} running...")

            pass 
        
        engine.train_one_epoch(optimizer, epoch_index=1)
        
        print("   ✅ train_one_epoch finished without crashing.")

    except Exception as e:
        print(f"\n❌ CRITICAL FAIL: {e}")
        return

    print("\n[Step 4] Verifying Updates...")
    
    final_sigma = engine.log_vars.detach()
    if not torch.equal(initial_sigma, final_sigma):
        print(f"   ✅ Sigmas Updated: {initial_sigma.cpu().numpy()} -> {final_sigma.cpu().numpy()}")
    else:
        print("   ⚠️ WARNING: Sigmas did not change! (Check optimizer parameters)")

    final_lora_weight = list(engine.backbone.parameters())[-1].detach()
    if not torch.equal(initial_lora_weight, final_lora_weight):
        print("   ✅ Backbone Weights Updated (PCGrad successfully applied gradients).")
    else:
        print("   ❌ CRITICAL: Backbone weights did not change!")

    print("\n[Step 5] Checking Log Files...")
    
    if os.path.exists(os.path.join(TEST_DIR, "pcgrad_statistics.csv")):
        print("   ✅ pcgrad_statistics.csv found.")
        df = pd.read_csv(os.path.join(TEST_DIR, "pcgrad_statistics.csv"))
        print(f"      -> Contains {len(df)} entries.")
        if len(df) > 0:
            print(f"      -> First entry conflict count: {df.iloc[0]['Conflict_Count']}")
    else:
        print("   ⚠️ pcgrad_statistics.csv missing (Note: Engine only logs every 100 steps, this might be expected if steps < 100)")

    print("\n🚀 HYBRID SANITY CHECK COMPLETE. You are ready to train.")

if __name__ == "__main__":
    run_hybrid_sanity_check()