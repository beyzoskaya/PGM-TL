from collections import defaultdict
import os
import sys
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

class TaskModel(nn.Module):
    
    def __init__(self, backbone, task_type, num_labels, task_id):
        super().__init__()
        self.backbone = backbone
        self.task_type = task_type
        self.num_labels = num_labels
        self.task_id = task_id
        self.output_dim = backbone.output_dim
        
        # Task head
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, num_labels)
        )
    
    def forward(self, batch):
        backbone_out = self.backbone(batch)
        
        if self.task_type == 'token_classification':
            features = backbone_out["residue_feature"]
        else:
            features = backbone_out["graph_feature"]
        
        logits = self.head(features)
        
        return {
            'logits': logits,
            'attention_mask': backbone_out.get("attention_mask"),
        }


class ModelsWrapper(nn.Module):
    
    def __init__(self, task_models, task_names):
        super().__init__()
        self.task_models = nn.ModuleList(task_models)
        self.task_names = task_names
    
    def forward(self, batches):
        """Forward pass through all task models"""
        all_loss = []
        all_metric = {}
        
        for task_id, batch in enumerate(batches):
            model = self.task_models[task_id]
            task_type = model.task_type
            
            outputs = model(batch)
            logits = outputs['logits']
            
            loss = self._compute_loss(logits, batch, task_type)
            all_loss.append(loss)
            
            metric = self._compute_metrics(logits, batch, task_type)
            
            for k, v in metric.items():
                metric_name = f"{self.task_names[task_id]} {k}"
                all_metric[metric_name] = v
        
        all_loss = torch.stack(all_loss)
        return all_loss, all_metric
    
    def _compute_loss(self, logits, batch, task_type):
        if 'targets' not in batch:
            raise ValueError("Batch missing 'targets'")
        
        targets_dict = batch['targets']
        if not isinstance(targets_dict, dict):
            raise ValueError(f"Expected targets to be dict, got {type(targets_dict)}")
        
        target_key = list(targets_dict.keys())[0]
        target = targets_dict[target_key]
        
        if task_type == 'token_classification':
            return self._loss_token_classification(logits, target, batch)
        elif task_type == 'regression':
            return self._loss_regression(logits, target)
        elif task_type == 'binary_classification':
            return self._loss_binary_classification(logits, target)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _loss_token_classification(self, logits, target, batch):
        device = logits.device
        
        if isinstance(target, list):
            tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
            target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(device)
        else:
            target_tensor = target.to(device)
        
        attention_mask = batch.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones((target_tensor.size(0), logits.size(1)), 
                                        dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)
        
        seq_len = logits.size(1)
        
        if target_tensor.size(1) > seq_len:
            target_tensor = target_tensor[:, :seq_len]
        elif target_tensor.size(1) < seq_len:
            pad = torch.full((target_tensor.size(0), seq_len - target_tensor.size(1)),
                             fill_value=-100, device=device, dtype=target_tensor.dtype)
            target_tensor = torch.cat([target_tensor, pad], dim=1)
        
        if attention_mask.size(1) > seq_len:
            attention_mask = attention_mask[:, :seq_len]
        elif attention_mask.size(1) < seq_len:
            pad = torch.ones((attention_mask.size(0), seq_len - attention_mask.size(1)),
                             dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, pad], dim=1)
        
        active = attention_mask.reshape(-1) == 1
        active_logits = logits.reshape(-1, logits.size(-1))[active]
        active_labels = target_tensor.reshape(-1)[active]
        
        if active_logits.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return F.cross_entropy(active_logits, active_labels)
    
    def _loss_regression(self, logits, target):
        device = logits.device
        
        if isinstance(target, list):
            target_tensor = torch.tensor(target, dtype=torch.float, device=device)
        else:
            target_tensor = target.to(device).float()
        
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        
        return F.mse_loss(logits, target_tensor)
    
    def _loss_binary_classification(self, logits, target):
        device = logits.device
        
        if isinstance(target, list):
            target_tensor = torch.tensor(target, dtype=torch.float, device=device)
        else:
            target_tensor = target.to(device).float()
        
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        
        return F.binary_cross_entropy_with_logits(logits, target_tensor)
    
    def _compute_metrics(self, logits, batch, task_type):
        if 'targets' not in batch:
            return {}
        
        targets_dict = batch['targets']
        target_key = list(targets_dict.keys())[0]
        target = targets_dict[target_key]
        
        if task_type == 'token_classification':
            return self._metrics_token_classification(logits, target, batch)
        elif task_type == 'regression':
            return self._metrics_regression(logits, target)
        elif task_type == 'binary_classification':
            return self._metrics_binary_classification(logits, target)
        else:
            return {}
    
    def _metrics_token_classification(self, logits, target, batch):
        device = logits.device
        
        if isinstance(target, list):
            tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
            target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(device)
        else:
            target_tensor = target.to(device)
        
        attention_mask = batch.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones((target_tensor.size(0), logits.size(1)), 
                                        dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)
        
        seq_len = logits.size(1)
        if target_tensor.size(1) > seq_len:
            target_tensor = target_tensor[:, :seq_len]
        elif target_tensor.size(1) < seq_len:
            pad = torch.full((target_tensor.size(0), seq_len - target_tensor.size(1)),
                             fill_value=-100, device=device, dtype=target_tensor.dtype)
            target_tensor = torch.cat([target_tensor, pad], dim=1)
        
        if attention_mask.size(1) > seq_len:
            attention_mask = attention_mask[:, :seq_len]
        elif attention_mask.size(1) < seq_len:
            pad = torch.ones((attention_mask.size(0), seq_len - attention_mask.size(1)),
                             dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, pad], dim=1)
        
        pred = logits.argmax(dim=-1)
        active = attention_mask.reshape(-1) == 1
        active_preds = pred.reshape(-1)[active]
        active_labels = target_tensor.reshape(-1)[active]
        
        acc = (active_preds == active_labels).float().mean().item() if active_preds.numel() > 0 else 0.0
        return {"accuracy": acc}
    
    def _metrics_regression(self, logits, target):
        device = logits.device
        
        if isinstance(target, list):
            target_tensor = torch.tensor(target, dtype=torch.float, device=device)
        else:
            target_tensor = target.to(device).float()
        
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        
        mse = F.mse_loss(logits, target_tensor).item()
        return {"mse": mse}
    
    def _metrics_binary_classification(self, logits, target):
        device = logits.device
        
        if isinstance(target, list):
            target_tensor = torch.tensor(target, dtype=torch.float, device=device)
        else:
            target_tensor = target.to(device).float()
        
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        
        pred = (torch.sigmoid(logits) > 0.5).long()
        acc = (pred == target_tensor.long()).float().mean().item()
        return {"accuracy": acc}


class MultiTaskEngine:
    """Training engine for multi-task learning"""
    
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets, 
                 optimizer, scheduler=None, batch_size=8, gradient_interval=1, 
                 num_worker=0, log_interval=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker
        self.log_interval = log_interval
        
        # Create task models
        task_models = []
        for task_id, config in enumerate(task_configs):
            model = TaskModel(
                backbone=backbone,
                task_type=config['type'],
                num_labels=config['num_labels'],
                task_id=task_id
            )
            task_models.append(model)
        
        # Wrap models
        task_names = [f"Task_{i}" for i in range(len(task_configs))]
        self.models = ModelsWrapper(task_models, task_names)
        self.models.to(self.device)
        
        self.train_sets = train_sets
        self.valid_sets = valid_sets
        self.test_sets = test_sets
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.epoch = 0
        self.step = 0
        
        # Loss normalization tracking
        self.loss_ema = None  # Exponential moving average of losses
        self.ema_decay = 0.9  # Higher = slower decay, more stable
        
        logger.info(f"Initialized MultiTaskEngine with {len(task_configs)} tasks on device {self.device}")
        for i, (train_set, valid_set, test_set) in enumerate(zip(train_sets, valid_sets, test_sets)):
            logger.info(f"Task {i}: Train={len(train_set)}, Valid={len(valid_set)}, Test={len(test_set)}")
    
    def collate_fn(self, batch):
        """Collate batch: handle variable-length sequences"""
        sequences = []
        targets_dict = defaultdict(list)
        
        for item in batch:
            if isinstance(item, dict):
                sequences.append(item['sequence'])
                targets = item.get('targets', {})
                if isinstance(targets, dict):
                    for key, value in targets.items():
                        targets_dict[key].append(value)
            else:
                sequences.append(getattr(item, 'sequence', ''))
        
        processed_targets = {}
        for key, values in targets_dict.items():
            if values and isinstance(values[0], list):
                processed_targets[key] = values
            else:
                try:
                    processed_targets[key] = torch.tensor(values, dtype=torch.float)
                except:
                    processed_targets[key] = values
        
        return {
            'sequence': sequences,
            'targets': processed_targets
        }
    
    def move_to_device(self, batch):
        if isinstance(batch, dict):
            moved_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    moved_batch[key] = value.to(self.device)
                elif isinstance(value, dict):
                    moved_batch[key] = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in value.items()
                    }
                else:
                    moved_batch[key] = value
            return moved_batch
        return batch
    
    def _update_loss_ema(self, per_task_loss):
        if self.loss_ema is None:
            # Initialize on first batch
            self.loss_ema = per_task_loss.detach().clone()
        else:
            # Update: new_ema = decay * old_ema + (1 - decay) * new_loss
            self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * per_task_loss.detach()
    
    def train(self, num_epoch=1, batch_per_epoch=None, tradeoff=1.0, weighting_strategy='center_task'):
        """
        Train for multiple epochs
        
        Args:
            num_epoch: number of epochs
            batch_per_epoch: batches per epoch (None = use minimum)
            tradeoff: weight for auxiliary tasks (only used with 'center_task' strategy)
            weighting_strategy: 'center_task', 'equal', or 'loss_norm'
        """
        logger.info(f"Starting training for {num_epoch} epochs with strategy={weighting_strategy}")
        if weighting_strategy == 'center_task':
            logger.info(f"  Center task weight: 1.0, Auxiliary weight: {tradeoff}")
        elif weighting_strategy == 'equal':
            logger.info(f"  All tasks equally weighted: 1.0")
        elif weighting_strategy == 'loss_norm':
            logger.info(f"  Loss normalization enabled (EMA decay={self.ema_decay})")
        
        self.models.train()
        
        for epoch in range(num_epoch):
            logger.info(f"Epoch {epoch + 1}/{num_epoch}")
            
            # Reset loss EMA each epoch
            self.loss_ema = None
            
            # Create dataloaders for this epoch
            dataloaders = []
            for train_set in self.train_sets:
                dataloader = DataLoader(
                    train_set,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_worker,
                    collate_fn=self.collate_fn,
                    pin_memory=True
                )
                dataloaders.append(iter(dataloader))
            
            # Determine batch_per_epoch
            if batch_per_epoch is None:
                lengths = [len(DataLoader(ts, batch_size=self.batch_size)) for ts in self.train_sets]
                batch_per_epoch = min(lengths)
            
            metrics_buffer = []
            progress_bar = tqdm(range(batch_per_epoch), desc=f"Epoch {epoch + 1}")
            
            for batch_id in progress_bar:
                batches = []
                
                # Get one batch per task
                for task_id, dataloader in enumerate(dataloaders):
                    try:
                        batch = next(dataloader)
                    except StopIteration:
                        dataloader = iter(DataLoader(
                            self.train_sets[task_id],
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_worker,
                            collate_fn=self.collate_fn,
                            pin_memory=True
                        ))
                        dataloaders[task_id] = dataloader
                        batch = next(dataloader)
                    
                    batch = self.move_to_device(batch)
                    batches.append(batch)
                
                # Forward and compute loss
                per_task_loss, metric = self.models(batches)
                
                # Update loss EMA for normalization
                if weighting_strategy == 'loss_norm':
                    self._update_loss_ema(per_task_loss)
                
                # Apply task weighting based on strategy
                if weighting_strategy == 'center_task':
                    weights = [1.0 if i == 0 else tradeoff for i in range(len(batches))]
                
                elif weighting_strategy == 'equal':
                    weights = [1.0 for i in range(len(batches))]
                
                elif weighting_strategy == 'loss_norm':
                    # Normalize losses by their EMA
                    # Add small epsilon to prevent division by zero
                    epsilon = 1e-8
                    normalized_losses = per_task_loss / (self.loss_ema + epsilon)
                    # Scale to have similar magnitude as original
                    normalized_losses = normalized_losses / normalized_losses.mean()
                    weights = normalized_losses.detach().cpu().tolist()
                    weights = [1.0 / w for w in weights]  # Inverse: high loss → lower weight
                    weights = [w / sum(weights) for w in weights]  # Normalize to sum to 1
                
                else:
                    raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")
                
                weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float)
                
                weighted_loss = (per_task_loss * weights_tensor).sum()
                weighted_loss = weighted_loss / self.gradient_interval
                weighted_loss.backward()
                
                metrics_buffer.append(metric)
                
                # Optimizer step
                if (batch_id + 1) % self.gradient_interval == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if len(metrics_buffer) > 0:
                        avg_metrics = self._average_metrics(metrics_buffer)
                        progress_bar.set_postfix(avg_metrics)
                    
                    metrics_buffer = []
                    self.step += 1
            
            self.epoch += 1
            
            if self.scheduler:
                self.scheduler.step()
    
    def _average_metrics(self, metrics_list):
        """Average metrics across accumulation steps"""
        if not metrics_list:
            return {}
        
        avg = defaultdict(float)
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    avg[key] += value
        
        for key in avg:
            avg[key] /= len(metrics_list)
        
        return dict(avg)
    
    @torch.no_grad()
    def evaluate(self, split='valid', log=True):
        """Evaluate on validation or test set"""
        logger.info(f"Evaluating on {split} set")
        
        test_sets = getattr(self, f"{split}_sets")
        self.models.eval()
        
        all_metrics = defaultdict(float)
        
        for task_id, (test_set, model) in enumerate(zip(test_sets, self.models.task_models)):
            dataloader = DataLoader(
                test_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_worker,
                collate_fn=self.collate_fn
            )
            
            task_metrics = defaultdict(list)
            
            for batch in tqdm(dataloader, desc=f"Evaluating Task {task_id}"):
                batch = self.move_to_device(batch)
                
                outputs = model(batch)
                logits = outputs['logits']
                
                metrics = self.models._compute_metrics(logits, batch, model.task_type)
                
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        task_metrics[key].append(value)
            
            for key, values in task_metrics.items():
                if values:
                    avg_value = sum(values) / len(values)
                    metric_name = f"Task_{task_id} {key}"
                    if task_id == 0:
                        metric_name = f"Center - {metric_name}"
                    all_metrics[metric_name] = avg_value
        
        if log:
            logger.info(f"{split.upper()} Results:")
            for key, value in all_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        return dict(all_metrics)
    
    def save(self, checkpoint_path):
        """Save checkpoint"""
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        state = {
            "epoch": self.epoch,
            "step": self.step,
            "optimizer": self.optimizer.state_dict(),
        }
        
        for i, model in enumerate(self.models.task_models):
            state[f"task_{i}"] = model.state_dict()
        
        torch.save(state, checkpoint_path)
    
    def load(self, checkpoint_path, load_optimizer=True):
        """Load checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        for i, model in enumerate(self.models.task_models):
            if f"task_{i}" in checkpoint:
                model.load_state_dict(checkpoint[f"task_{i}"])
        
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)