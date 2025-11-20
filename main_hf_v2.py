import torch
import numpy as np
import os
from torch.optim import AdamW

from protbert_hf import SharedProtBert
from engine_hf_with_task_specific_encoder import MultiTaskEngine
from flip_hf import Thermostability, SecondaryStructure, CloningCLF

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 16
SANITY_CHECK = True
SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/dynamic_weighting"
DEBUG_INTERVAL = 100

EPOCHS = 2

# Learning rate
LR = 1e-4

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def main():
    set_seed(SEED)
    save_dir = ensure_dir(SAVE_DIR)
    
    # Load backbone
    backbone = SharedProtBert(
        model_name="Rostlab/prot_bert_bfd",
        lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        freeze_backbone=True,
        verbose=True
    )

    # Task configurations
    task_configs = [
        {"name": "thermo", "type": "regression", "num_labels": 1},
        {"name": "ssp", "type": "per_residue_classification", "num_labels": 8},
        {"name": "clf", "type": "sequence_classification", "num_labels": 2}
    ]

    # Load datasets
    train_sets = [Thermostability(split="train"), SecondaryStructure(split="train"), CloningCLF(split="train")]
    valid_sets = [Thermostability(split="valid"), SecondaryStructure(split="valid"), CloningCLF(split="valid")]
    test_sets = [Thermostability(split="test"), SecondaryStructure(split="test"), CloningCLF(split="test")]

    # Initialize engine
    engine = MultiTaskEngine(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=train_sets,
        valid_sets=valid_sets,
        test_sets=test_sets,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        save_dir=save_dir,
        verbose=True
    )

    if SANITY_CHECK:
        print("[SANITY CHECK] Running one batch per task...")
        engine.train_one_epoch(optimizer=AdamW(list(backbone.parameters()), lr=LR), max_batches_per_loader=1)
        print("[SANITY CHECK] Done. Everything works!")
        return

    # Optimizer
    optimizer = AdamW(list(backbone.parameters()) + list(engine.task_heads.parameters()), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        avg_loss = engine.train_one_epoch(optimizer)
        print(f"[EPOCH] Average training loss: {avg_loss:.4f}")

        # Validation
        val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=epoch)

        # Print best so far
        engine.print_best_metrics()

        # Save model + PEFT weights after each epoch
        epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
        ensure_dir(epoch_save_dir)
        backbone.model.save_pretrained(epoch_save_dir)
        for i, head in enumerate(engine.task_heads):
            torch.save(head.state_dict(), os.path.join(epoch_save_dir, f"task_head_{i}.pt"))
        print(f"[SAVE] Saved backbone + task heads to {epoch_save_dir}")

    # Test evaluation
    print("\n[TEST] Evaluating on test sets...")
    test_metrics = engine.evaluate(engine.test_loaders, split_name="Test")
    print("[DONE] Training + evaluation completed!")

if __name__ == "__main__":
    main()
