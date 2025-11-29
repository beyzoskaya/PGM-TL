import os
import torch
import numpy as np
import csv
from transformers import get_linear_schedule_with_warmup

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_prompt import TaskPromptedEngine

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16        
EPOCHS = 5 
LEARNING_RATE = 1e-4

UNFROZEN_LAYERS = 2 
LORA_RANK = 16

SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/framework_prompt_pcgrad"
os.makedirs(SAVE_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalize_regression_targets(train_ds, valid_ds, test_ds):
    print("\n[Data] Normalizing Thermostability targets...")
    full_dataset = train_ds.dataset; train_indices = train_ds.indices
    all_raw = full_dataset.targets['target']
    vals = [all_raw[i] for i in train_indices if all_raw[i] is not None]
    mean = np.mean(vals); std = np.std(vals)
    new_t = []
    for t in all_raw:
        if t is None: new_t.append(None)
        else: new_t.append((t-mean)/std)
    full_dataset.targets['target'] = new_t
    print("  ✓ Normalization complete.")

def main():
    set_seed(SEED)
    print(f"🚀 Running FINAL FRAMEWORK: Prompt-Tuning + PCGrad on {DEVICE}")
    print(f"   (Unfrozen Layers: {UNFROZEN_LAYERS})")

    # 1. DATA
    print("[1/5] Loading Datasets...")
    ds_t = Thermostability(verbose=0); t_train, t_val, t_test = ds_t.split()
    normalize_regression_targets(t_train, t_val, t_test)
    ds_s = SecondaryStructure(verbose=0); s_train, s_val, s_test = ds_s.split()
    ds_c = CloningCLF(verbose=0); c_train, c_val, c_test = ds_c.split()

    train_sets = [t_train, s_train, c_train]
    valid_sets = [t_val, s_val, c_val]
    test_sets  = [t_test, s_test, c_test]

    task_configs = [
        {'name': 'Thermostability', 'type': 'regression', 'num_labels': 1},
        {'name': 'SecStructure', 'type': 'token_classification', 'num_labels': 8},
        {'name': 'Cloning', 'type': 'sequence_classification', 'num_labels': 2}
    ]

    # 2. MODEL
    print("[2/5] Initializing Backbone...")
    backbone = SharedProtBert(
        model_name="Rostlab/prot_bert_bfd",
        lora_rank=LORA_RANK,
        lora_dropout=0.1,
        unfrozen_layers=UNFROZEN_LAYERS 
    )

    # 3. ENGINE
    print("[3/5] Setting up Prompted Engine...")
    engine = TaskPromptedEngine(
        backbone=backbone, task_configs=task_configs,
        train_sets=train_sets, valid_sets=valid_sets, test_sets=test_sets,
        batch_size=BATCH_SIZE, device=DEVICE, save_dir=SAVE_DIR
    )

    # VERIFY PROMPTS
    print(f"   [Check] Number of Prompts: {len(engine.task_prompts)}")
    print(f"   [Check] Prompt Shape: {engine.task_prompts[0].shape}")

    optimizer = torch.optim.AdamW(engine.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    max_len = max([len(l) for l in engine.train_loaders])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*max_len*EPOCHS), num_training_steps=max_len*EPOCHS)

    # 4. LOGGING
    history_path = os.path.join(SAVE_DIR, "training_history.csv")
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ["Epoch", "Train_Combined_Loss"] + [f"Val_{t['name']}" for t in task_configs]
        writer.writerow(headers)

    best_val_thermo_loss = float('inf')

    # 5. LOOP
    for epoch in range(EPOCHS):
        print(f"\n{'='*30} EPOCH {epoch+1}/{EPOCHS} {'='*30}")
        
        # Train
        train_res = engine.train_one_epoch(optimizer, scheduler, epoch+1)
        print(f"\n  [TRAIN] Combined Loss: {train_res['avg_loss']:.4f}")

        # Evaluate
        metrics = engine.evaluate(loader_list=engine.valid_loaders, split_name="VALIDATION")
        engine.evaluate(loader_list=engine.test_loaders, split_name="TEST")

        # Save Logic
        row = [epoch+1, train_res['avg_loss']]
        for t in task_configs:
            val_str = metrics.get(t['name'], "0.0")
            try: row.append(float(val_str.split(":")[-1].strip()))
            except: row.append(0.0)
        with open(history_path, 'a', newline='') as f: csv.writer(f).writerow(row)

        torch.save(engine.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt"))
        
        try:
            curr_mse = float(metrics['Thermostability'].split(":")[1])
            if curr_mse < best_val_thermo_loss:
                best_val_thermo_loss = curr_mse
                torch.save(engine.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
                print(f"New Best Prompt Model Saved! (Thermo MSE: {best_val_thermo_loss:.4f})")
        except: pass

    print("\nDone!")

if __name__ == "__main__":
    main()