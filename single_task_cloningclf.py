import os
import sys
import logging
import argparse
import numpy as np
import yaml
from easydict import EasyDict
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset

from flip_hf import CloningCLF
from protbert_hf import ProtBertWithLoRA
from engine_hf import ModelsWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=4, help="number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--use_drive", action="store_true", help="save outputs to Google Drive")
    parser.add_argument("--output_dir", type=str, default="./binary_clf_outputs")
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    fh = logging.FileHandler(os.path.join(output_dir, "training.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def setup_drive_output():
    base_path = "/content/drive/MyDrive/protein_multitask_outputs/single_task_outputs"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created on Drive: {output_dir}")
    return output_dir

def collate_fn(batch):

    sequences = []
    targets = []
    
    for item in batch:
        sequences.append(item['sequence'])
        if isinstance(item['targets'], dict):
            targets.append(item['targets'].get('target', 0.0))
        else:
            targets.append(item['targets'])
    
    return {
        'sequence': sequences,
        'targets': torch.tensor(targets, dtype=torch.float),
        'task_type': 'binary_classification'
    }

def train_epoch(model, train_loader, optimizer, device, logger):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        sequences = batch['sequence']
        targets = batch['targets'].to(device)
        
        # Forward pass
        outputs = model({
            'sequence': sequences,
            'targets': {'target': targets.cpu()},
            'task_type': 'binary_classification'
        })
        
        logits = outputs['logits']
        
        # Compute loss
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits_flat = logits.squeeze(-1)
        else:
            logits_flat = logits.view(-1)
        
        loss = nn.functional.binary_cross_entropy_with_logits(logits_flat, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss

@torch.no_grad()
def evaluate(model, eval_loader, device, logger, split_name="Validation"):

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    for batch in eval_loader:
        sequences = batch['sequence']
        targets = batch['targets'].to(device)
        
        outputs = model({
            'sequence': sequences,
            'targets': {'target': targets.cpu()},
            'task_type': 'binary_classification'
        })
        
        logits = outputs['logits']
        
        # Compute loss
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits_flat = logits.squeeze(-1)
        else:
            logits_flat = logits.view(-1)
        
        loss = nn.functional.binary_cross_entropy_with_logits(logits_flat, targets)
        total_loss += loss.item() * targets.size(0)
        
        # Compute predictions
        preds = (torch.sigmoid(logits_flat) > 0.5).long()
        targets_long = targets.long()
        
        total_correct += (preds == targets_long).sum().item()
        total_samples += targets.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets_long.cpu().numpy())
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    logger.info(f"{split_name} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'preds': all_preds,
        'targets': all_targets
    }

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device(args.device)
    
    if args.use_drive:
        output_dir = setup_drive_output()
    else:
        output_dir = args.output_dir
    
    logger = setup_logger(output_dir)
    os.chdir(output_dir) 
    
    logger.info("="*80)
    logger.info("BINARY CLASSIFICATION BASELINE - ProtBert on Cloning Dataset")
    logger.info("="*80)
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}, Learning rate: {args.lr}, Epochs: {args.num_epoch}")
    
    # Load dataset
    logger.info("\nLoading CloningCLF dataset...")
    dataset = CloningCLF(path="./data", verbose=1)
    train_set, valid_set, test_set = dataset.split()
    
    logger.info(f"Dataset splits - Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")
    
    # Example data point
    example_idx = 0
    if len(train_set) > example_idx:
        example = train_set[example_idx]
        logger.info(f"\nExample data point:")
        logger.info(f"  Sequence: {example['sequence'][:60]}...")
        logger.info(f"  Target: {example['targets']}")
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    
    # Create model
    logger.info("\nCreating ProtBertWithLoRA model for binary classification...")
    model = ProtBertWithLoRA(
        model_name="Rostlab/prot_bert_bfd",
        num_labels=1,
        readout="pooler",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type="binary_classification"
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,} "
                f"({trainable_params/total_params*100:.2f}%)")
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("TRAINING PHASE")
    logger.info("="*80)
    
    best_val_acc = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    
    for epoch in range(args.num_epoch):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epoch}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, logger)
        logger.info(f"Training - Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = evaluate(model, valid_loader, device, logger, "Validation")
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✓ New best model saved (accuracy: {best_val_acc:.4f})")
    
    # Load best model and evaluate on test
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80)
    
    model.load_state_dict(torch.load(best_model_path))
    logger.info(f"Loaded best model from epoch {best_epoch}")
    
    val_metrics = evaluate(model, valid_loader, device, logger, "Validation (Best Model)")
    test_metrics = evaluate(model, test_loader, device, logger, "Test (Best Model)")
    
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Model saved to: {best_model_path}")
    
    return model, test_metrics

if __name__ == "__main__":
    model, metrics = main()