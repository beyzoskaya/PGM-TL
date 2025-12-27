import torch
import numpy as np
import os
from torch.utils.data import DataLoader

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engines.engine_hf_with_task_specific_encoder import MultiTaskEngine, multitask_collate_fn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_header(msg):
    print(f"\n" + "="*60)
    print(f"DIAGNOSE: {msg}")
    print("="*60)

def diagnose():
    print_header(f"System Check - Device: {DEVICE}")

    # =================================================================
    # 1. DATASET LOADING & TARGET INSPECTION
    # =================================================================
    print_header("1. Loading Datasets & Checking Raw Types")
    
    # Load subsets
    ds_thermo = Thermostability(verbose=0)
    ds_ssp = SecondaryStructure(verbose=0)
    ds_clf = CloningCLF(verbose=0)

    # Check Thermostability Raw
    sample_thermo = ds_thermo[0]
    val_thermo = sample_thermo['targets']['target']
    print(f"[Thermo] Raw sample: {val_thermo}")
    if not isinstance(val_thermo, float):
        print("  ❌ ERROR: Thermo target is not a float!")
    else:
        print("  ✅ Type Check Passed (Float)")

    # Check SSP Raw
    sample_ssp = ds_ssp[0]
    val_ssp = sample_ssp['targets']['target']
    print(f"[SSP]    Raw sample length: {len(val_ssp)} (Seq len: {len(sample_ssp['sequence'])})")
    if not isinstance(val_ssp, list):
        print("  ❌ ERROR: SSP target is not a list!")
    else:
        print("  ✅ Type Check Passed (List)")

    # Check Cloning Raw
    sample_clf = ds_clf[0]
    val_clf = sample_clf['targets']['target']
    print(f"[Clone]  Raw sample: {val_clf}")
    if not isinstance(val_clf, int):
        print("  ❌ ERROR: Cloning target is not an int!")
    else:
        print("  ✅ Type Check Passed (Int)")

    # =================================================================
    # 2. NORMALIZATION CHECK
    # =================================================================
    print_header("2. Verifying Thermostability Normalization")
    
    # Extract raw values
    raw_values = [x['targets']['target'] for x in ds_thermo if x['targets']['target'] is not None]
    raw_mean = np.mean(raw_values)
    raw_std = np.std(raw_values)
    print(f"  Raw Mean: {raw_mean:.4f} | Raw Std: {raw_std:.4f}")
    
    # Apply Normalization (Simulating main.py logic)
    print("  >> Applying Normalization...")
    for i in range(len(ds_thermo.sequences)):
        t = ds_thermo.targets['target'][i]
        if t is not None:
            ds_thermo.targets['target'][i] = (t - raw_mean) / raw_std
            
    # Check new values
    new_values = [x['targets']['target'] for x in ds_thermo if x['targets']['target'] is not None]
    new_mean = np.mean(new_values)
    new_std = np.std(new_values)
    print(f"  New Mean: {new_mean:.4f} | New Std: {new_std:.4f}")
    
    if abs(new_mean) < 1e-5 and abs(new_std - 1.0) < 1e-5:
        print("  ✅ Normalization Successful (Mean~0, Std~1)")
    else:
        print("  ❌ WARNING: Normalization logic might be off.")

    # =================================================================
    # 3. COLLATE & PADDING CHECK
    # =================================================================
    print_header("3. Testing Collate & Padding Logic")
    
    # Initialize Tokenizer via Model class (lightweight)
    print("  Initializing Tokenizer...")
    backbone = SharedProtBert(lora_rank=8, unfrozen_layers=0) # Lightweight init
    tokenizer = backbone.tokenizer
    
    # Create a Mixed Batch: One short SSP, One Long SSP
    # We take two samples from SSP dataset
    batch_samples = [ds_ssp[0], ds_ssp[1]] 
    
    print(f"  Sample 1 Length: {len(batch_samples[0]['sequence'])}")
    print(f"  Sample 2 Length: {len(batch_samples[1]['sequence'])}")
    
    collated = multitask_collate_fn(batch_samples, tokenizer)
    
    inp_shape = collated['input_ids'].shape
    tgt_shape = collated['targets'].shape
    
    print(f"  Collated Input Shape: {inp_shape}")
    print(f"  Collated Target Shape: {tgt_shape}")
    
    if inp_shape[1] == tgt_shape[1]:
        print("  ✅ Sequence Lengths Match")
    else:
        print("  ❌ ERROR: Mismatch between Input and Target lengths!")

    # Check for -100 padding
    # The shorter sequence should be padded with -100 at the end
    print("  Checking padding values in target...")
    found_ignore_index = (collated['targets'] == -100).any()
    if found_ignore_index:
        print("  ✅ Found -100 in targets (Padding logic works)")
    else:
        print("  ⚠️ WARNING: No -100 found (Maybe sequences were same length?)")

    # =================================================================
    # 4. MODEL & ENGINE BUILD
    # =================================================================
    print_header("4. Building Model & Engine")
    
    task_configs = [
        {'name': 'Thermo', 'type': 'regression', 'num_labels': 1},
        {'name': 'SSP', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Clone', 'type': 'sequence_classification', 'num_labels': 2}
    ]
    
    engine = MultiTaskEngine(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=[ds_thermo], # Dummy sets just to init
        valid_sets=None,
        device=DEVICE
    )
    
    # Check Gradients
    trainable_params = sum(p.numel() for p in engine.parameters() if p.requires_grad)
    print(f"  Total Trainable Parameters: {trainable_params:,}")
    if trainable_params > 0:
        print("  ✅ Model has trainable parameters")
    else:
        print("  ❌ ERROR: Model is completely frozen!")

    # =================================================================
    # 5. FORWARD PASS & LOSS CHECK
    # =================================================================
    print_header("5. Running Forward Pass (One Batch Per Task)")
    
    engine.train()
    
    # Test Thermo
    print("\n  [Testing Thermostability Forward]")
    batch_thermo = multitask_collate_fn([ds_thermo[0], ds_thermo[1]], tokenizer)
    inputs = batch_thermo['input_ids'].to(DEVICE)
    mask = batch_thermo['attention_mask'].to(DEVICE)
    targets = batch_thermo['targets'].to(DEVICE)
    
    emb = engine.backbone(inputs, mask, task_type='sequence')
    preds = engine.heads[0](emb)
    loss = engine.loss_fns[0](preds, targets)
    print(f"    > Input Shape: {inputs.shape}")
    print(f"    > Logits Shape: {preds.shape}")
    print(f"    > Loss: {loss.item():.4f}")
    if torch.isnan(loss): print("    ❌ LOSS IS NAN"); exit()
    else: print("    ✅ Forward pass successful")

    # Test SSP
    print("\n  [Testing Secondary Structure Forward]")
    batch_ssp = multitask_collate_fn([ds_ssp[0], ds_ssp[1]], tokenizer)
    inputs = batch_ssp['input_ids'].to(DEVICE)
    mask = batch_ssp['attention_mask'].to(DEVICE)
    targets = batch_ssp['targets'].to(DEVICE)
    
    emb = engine.backbone(inputs, mask, task_type='token')
    preds = engine.heads[1](emb)
    # View logic matches engine
    loss = engine.loss_fns[1](preds.view(-1, 8), targets.view(-1)) 
    print(f"    > Input Shape: {inputs.shape}")
    print(f"    > Logits Shape: {preds.shape}")
    print(f"    > Loss: {loss.item():.4f}")
    if torch.isnan(loss): print("    ❌ LOSS IS NAN"); exit()
    else: print("    ✅ Forward pass successful")

    # Test Cloning
    print("\n  [Testing Cloning Forward]")
    batch_clf = multitask_collate_fn([ds_clf[0], ds_clf[1]], tokenizer)
    inputs = batch_clf['input_ids'].to(DEVICE)
    mask = batch_clf['attention_mask'].to(DEVICE)
    targets = batch_clf['targets'].to(DEVICE)
    
    emb = engine.backbone(inputs, mask, task_type='sequence')
    preds = engine.heads[2](emb)
    loss = engine.loss_fns[2](preds, targets)
    print(f"    > Input Shape: {inputs.shape}")
    print(f"    > Logits Shape: {preds.shape}")
    print(f"    > Loss: {loss.item():.4f}")
    if torch.isnan(loss): print("    ❌ LOSS IS NAN"); exit()
    else: print("    ✅ Forward pass successful")

    print_header("DIAGNOSIS COMPLETE - ALL SYSTEMS GO")

if __name__ == "__main__":
    diagnose()