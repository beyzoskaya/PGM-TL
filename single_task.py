import os
import sys
import time
import logging
import argparse
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from easydict import EasyDict
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import ProtBertWithLoRA
import json
from datetime import datetime
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/single_task_same_configs"
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = logging.getLogger("")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

log_path = os.path.join(SAVE_DIR, f"single_task_run_{timestamp}.log")
fh = logging.FileHandler(log_path)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info(f"Logging to {log_path}")

class SingleTaskModel(nn.Module):
    
    def __init__(self, backbone, task_type, num_labels, output_dim=768):
        super().__init__()
        self.backbone = backbone
        self.task_type = task_type
        self.num_labels = num_labels
        
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(output_dim, num_labels)
        )
    
    def forward(self, batch):
        backbone_out = self.backbone(batch)
        
        if self.task_type == 'token_classification':
            features = backbone_out["residue_feature"]  # [batch, seq_len, hidden]
        else:
            features = backbone_out["graph_feature"]  # [batch, hidden]
        
        logits = self.head(features)
        
        return {
            'logits': logits,
            'attention_mask': backbone_out.get("attention_mask"),
        }


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataset(dataset_type):
    """Create and split dataset"""
    if dataset_type == 'Thermostability':
        dataset = Thermostability(path='./data', verbose=1)
    elif dataset_type == 'SecondaryStructure':
        dataset = SecondaryStructure(path='./data', verbose=1)
    elif dataset_type == 'CloningCLF':
        dataset = CloningCLF(path='./data', verbose=1)
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")
    
    train_set, valid_set, test_set = dataset.split()
    return train_set, valid_set, test_set


def collate_fn(batch):
    sequences = []
    targets_list = []
    
    for item in batch:
        sequences.append(item['sequence'])
        targets = item.get('targets', {})
        if isinstance(targets, dict):
            targets_list.append(targets)
    
    # Extract targets
    all_targets = {}
    for targets in targets_list:
        for key, value in targets.items():
            if key not in all_targets:
                all_targets[key] = []
            all_targets[key].append(value)
    
    # Convert targets to appropriate format
    processed_targets = {}
    for key, values in all_targets.items():
        if values and isinstance(values[0], list):
            # Token-level: keep as list of lists
            processed_targets[key] = values
        else:
            # Sequence-level: convert to tensor
            try:
                processed_targets[key] = torch.tensor(values, dtype=torch.float)
            except:
                processed_targets[key] = values
    
    return {
        'sequence': sequences,
        'targets': processed_targets
    }


def move_to_device(batch, device):
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, dict):
            moved_batch[key] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in value.items()
            }
        else:
            moved_batch[key] = value
    return moved_batch


def compute_loss(model, batch, device):
    
    outputs = model(batch)
    logits = outputs['logits']
    
    # Get targets
    targets_dict = batch['targets']
    target_key = list(targets_dict.keys())[0]
    target = targets_dict[target_key]
    
    task_type = model.task_type
    
    # Token-level classification
    if task_type == 'token_classification':
        if isinstance(target, list):
            tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
            target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(device)
        else:
            target_tensor = target.to(device)
        
        attention_mask = outputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones((target_tensor.size(0), logits.size(1)), 
                                       dtype=torch.long, device=device)
        
        seq_len = logits.size(1)
        if target_tensor.size(1) > seq_len:
            target_tensor = target_tensor[:, :seq_len]
        elif target_tensor.size(1) < seq_len:
            pad = torch.full((target_tensor.size(0), seq_len - target_tensor.size(1)),
                           fill_value=-100, device=device, dtype=target_tensor.dtype)
            target_tensor = torch.cat([target_tensor, pad], dim=1)
        
        active = attention_mask.reshape(-1) == 1
        active_logits = logits.reshape(-1, logits.size(-1))[active]
        active_labels = target_tensor.reshape(-1)[active]
        
        if active_logits.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return F.cross_entropy(active_logits, active_labels)
    
    # Regression
    elif task_type == 'regression':
        if isinstance(target, list):
            target_tensor = torch.tensor(target, dtype=torch.float, device=device)
        else:
            target_tensor = target.to(device).float()
        
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        
        return F.mse_loss(logits, target_tensor)
    
    # Binary classification
    elif task_type == 'binary_classification':
        if isinstance(target, list):
            target_tensor = torch.tensor(target, dtype=torch.float, device=device)
        else:
            target_tensor = target.to(device).float()
        
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        
        return F.binary_cross_entropy_with_logits(logits, target_tensor)


def compute_metrics(model, batch, device):

    outputs = model(batch)
    logits = outputs['logits']
    
    targets_dict = batch['targets']
    target_key = list(targets_dict.keys())[0]
    target = targets_dict[target_key]
    
    task_type = model.task_type
    
    # Token-level classification
    if task_type == 'token_classification':
        if isinstance(target, list):
            tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
            target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(device)
        else:
            target_tensor = target.to(device)
        
        attention_mask = outputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones((target_tensor.size(0), logits.size(1)), 
                                       dtype=torch.long, device=device)
        
        seq_len = logits.size(1)
        if target_tensor.size(1) > seq_len:
            target_tensor = target_tensor[:, :seq_len]
        elif target_tensor.size(1) < seq_len:
            pad = torch.full((target_tensor.size(0), seq_len - target_tensor.size(1)),
                           fill_value=-100, device=device, dtype=target_tensor.dtype)
            target_tensor = torch.cat([target_tensor, pad], dim=1)
        
        pred = logits.argmax(dim=-1)
        active = attention_mask.reshape(-1) == 1
        active_preds = pred.reshape(-1)[active]
        active_labels = target_tensor.reshape(-1)[active]
        
        acc = (active_preds == active_labels).float().mean().item() if active_preds.numel() > 0 else 0.0
        return {"accuracy": acc}
    
    # Regression
    elif task_type == 'regression':
        if isinstance(target, list):
            target_tensor = torch.tensor(target, dtype=torch.float, device=device)
        else:
            target_tensor = target.to(device).float()
        
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        
        mse = F.mse_loss(logits, target_tensor).item()
        return {"mse": mse}
    
    # Binary classification
    elif task_type == 'binary_classification':
        if isinstance(target, list):
            target_tensor = torch.tensor(target, dtype=torch.float, device=device)
        else:
            target_tensor = target.to(device).float()
        
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        
        pred = (torch.sigmoid(logits) > 0.5).long()
        acc = (pred == target_tensor.long()).float().mean().item()
        return {"accuracy": acc}


def train_single_task(task_name, task_config, num_epochs=4, batch_size=8, 
                     gradient_interval=6, learning_rate=3e-5, weight_decay=0.01):
    """Train a single task model"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {task_name}")
    logger.info(f"{'='*60}")
    
    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    logger.info(f"Loading dataset: {task_name}")
    train_set, valid_set, test_set = create_dataset(task_name)
    
    # Create model
    logger.info("Creating ProtBertWithLoRA model")
    backbone = ProtBertWithLoRA(
        model_name='Rostlab/prot_bert_bfd',
        readout='mean',
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    model = SingleTaskModel(
        backbone=backbone,
        task_type=task_config['type'],
        num_labels=task_config['num_labels'],
        output_dim=backbone.output_dim
    )
    model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info(f"Backbone: Total={total_params}, Trainable={trainable_params} ({trainable_params/total_params*100:.1f}%)")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Training loop
    best_metric = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        model.train()
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        metrics_buffer = []
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = move_to_device(batch, device)
            
            loss = compute_loss(model, batch, device)
            loss = loss / gradient_interval
            loss.backward()
            
            metric = compute_metrics(model, batch, device)
            metrics_buffer.append(metric)
            
            if (batch_idx + 1) % gradient_interval == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                if len(metrics_buffer) > 0:
                    avg_metrics = {}
                    for m in metrics_buffer:
                        for k, v in m.items():
                            if k not in avg_metrics:
                                avg_metrics[k] = []
                            avg_metrics[k].append(v)
                    
                    avg_metrics = {k: sum(v) / len(v) for k, v in avg_metrics.items()}
                    progress_bar.set_postfix(avg_metrics)
                
                metrics_buffer = []
        
        scheduler.step()
        
        # Validate
        model.eval()
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn
        )
        
        valid_metrics = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                batch = move_to_device(batch, device)
                metric = compute_metrics(model, batch, device)
                valid_metrics.append(metric)
        
        avg_valid = {}
        for m in valid_metrics:
            for k, v in m.items():
                if k not in avg_valid:
                    avg_valid[k] = []
                avg_valid[k].append(v)
        
        avg_valid = {k: sum(v) / len(v) for k, v in avg_valid.items()}
        logger.info(f"Validation: {avg_valid}")
        
        # Track best
        metric_key = list(avg_valid.keys())[0]
        if avg_valid[metric_key] < best_metric:
            best_metric = avg_valid[metric_key]
            best_epoch = epoch
            ckpt_path = os.path.join(SAVE_DIR, f"{task_name}_best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved new best model for {task_name} → {ckpt_path}")
    
    # Test on best epoch
    logger.info(f"\nTesting (epoch {best_epoch + 1}):")
    model.eval()
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )
    
    test_metrics = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = move_to_device(batch, device)
            metric = compute_metrics(model, batch, device)
            test_metrics.append(metric)
    
    avg_test = {}
    for m in test_metrics:
        for k, v in m.items():
            if k not in avg_test:
                avg_test[k] = []
            avg_test[k].append(v)
    
    avg_test = {k: sum(v) / len(v) for k, v in avg_test.items()}
    logger.info(f"Test Results: {avg_test}")
    
    return avg_test

def main():
    tasks = {
        'SecondaryStructure': {'type': 'token_classification', 'num_labels': 8},
        'Thermostability': {'type': 'regression', 'num_labels': 1},
        'CloningCLF': {'type': 'binary_classification', 'num_labels': 1}
    }
    num_epochs = 4
    batch_size = 8
    gradient_interval = 6
    learning_rate = 3e-5
    weight_decay = 0.01
    results = {}
    for task_name, task_config in tasks.items():
        result = train_single_task(task_name, task_config, num_epochs, batch_size, gradient_interval, learning_rate, weight_decay)
        results[task_name] = result

    logger.info(f"\n{'='*60}")
    logger.info("SINGLE-TASK TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    for task_name, result in results.items():
        logger.info(f"{task_name}: {result}")

    result_path = os.path.join(SAVE_DIR, f"results_{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to: {result_path}")

if __name__ == "__main__":
    main()