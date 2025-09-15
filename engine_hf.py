from collections import defaultdict
import os
import sys
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelsWrapper(nn.Module):
    """
    Wrapper of multiple task models for multi-task learning.
    """

    def __init__(self, models, names):
        super(ModelsWrapper, self).__init__()
        self.models = nn.ModuleList(models)
        self.names = names

    def forward(self, batches):
        all_loss = []
        all_metric = defaultdict(float)
        
        for id, batch in enumerate(batches):
            if hasattr(self.models[id], 'compute_loss'):
                loss, metric = self.models[id].compute_loss(batch)
            else:
                # Default forward pass
                outputs = self.models[id](batch)
                loss = self.compute_default_loss(outputs, batch)
                metric = self.compute_default_metrics(outputs, batch)
            
            for k, v in metric.items():
                name = self.names[id] + " " + k
                if id == 0:
                    name = "Center - " + name
                all_metric[name] = v
            all_loss.append(loss)
        
        all_loss = torch.stack(all_loss)
        return all_loss, all_metric
    
    def compute_default_loss(self, outputs, batch):
        """Compute default loss for classification tasks"""
        logits = outputs["logits"]
        
        # Extract targets
        if isinstance(batch, dict) and 'targets' in batch:
            targets = batch['targets']
            if isinstance(targets, dict):
                # Use first target as default
                target_key = list(targets.keys())[0]
                target = targets[target_key]
            else:
                target = targets
        else:
            raise ValueError("Cannot find targets in batch")
        
        # Handle different target types
        if isinstance(target, list):
            target = torch.tensor(target, dtype=torch.float)
        
        target = target.to(logits.device)
        
        # Compute loss based on problem type
        if logits.size(-1) == 1:
            # Regression
            loss = F.mse_loss(logits.squeeze(), target.float())
        else:
            # Classification
            loss = F.cross_entropy(logits, target.long())
        
        return loss
    
    def compute_default_metrics(self, outputs, batch):
        """Compute default metrics"""
        logits = outputs["logits"]
        
        # Extract targets
        if isinstance(batch, dict) and 'targets' in batch:
            targets = batch['targets']
            if isinstance(targets, dict):
                target_key = list(targets.keys())[0]
                target = targets[target_key]
            else:
                target = targets
        else:
            return {"loss": 0.0}
        
        if isinstance(target, list):
            target = torch.tensor(target, dtype=torch.float)
        
        target = target.to(logits.device)
        
        metrics = {}
        
        if logits.size(-1) == 1:
            # Regression metrics
            mse = F.mse_loss(logits.squeeze(), target.float())
            metrics["mse"] = mse.item()
            metrics["rmse"] = torch.sqrt(mse).item()
        else:
            # Classification metrics
            pred = torch.argmax(logits, dim=-1)
            acc = (pred == target.long()).float().mean()
            metrics["accuracy"] = acc.item()
        
        return metrics

    def __getitem__(self, id):
        return self.models[id]


class MultiTaskEngine:
    """
    Multi-Task Learning engine compatible with HuggingFace datasets and models.
    """

    def __init__(self, tasks, train_sets, valid_sets, test_sets, optimizer, scheduler=None, 
                 batch_size=1, gradient_interval=1, num_worker=0, log_interval=100,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker
        self.log_interval = log_interval
        
        # Wrap models
        task_names = [f"Task_{i}" for i in range(len(tasks))]
        self.models = ModelsWrapper(tasks, names=task_names)
        self.models.to(self.device)
        
        self.train_sets = train_sets
        self.valid_sets = valid_sets
        self.test_sets = test_sets
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.epoch = 0
        self.step = 0
        
        logger.info(f"Initialized MultiTaskEngine with {len(tasks)} tasks")
        for i, (train_set, valid_set, test_set) in enumerate(zip(train_sets, valid_sets, test_sets)):
            logger.info(f"Task {i}: Train={len(train_set)}, Valid={len(valid_set)}, Test={len(test_set)}")

    def collate_fn(self, batch):
        """Custom collate function for protein sequences"""
        sequences = []
        targets_list = []
        
        for item in batch:
            if isinstance(item, dict):
                sequences.append(item['sequence'])
                targets_list.append(item.get('targets', {}))
            else:
                # Handle other formats
                sequences.append(getattr(item, 'sequence', ''))
                targets_list.append({})
        
        # Combine targets
        combined_targets = defaultdict(list)
        for targets in targets_list:
            if isinstance(targets, dict):
                for key, value in targets.items():
                    combined_targets[key].append(value)
        
        # Convert to tensors
        target_tensors = {}
        for key, values in combined_targets.items():
            try:
                target_tensors[key] = torch.tensor(values, dtype=torch.float)
            except:
                target_tensors[key] = values  # Keep as list if conversion fails
        
        return {
            'sequence': sequences,
            'targets': target_tensors
        }

    def train(self, num_epoch=1, batch_per_epoch=None, tradeoff=1.0):
        """
        Train the model for multiple epochs.
        """
        logger.info(f"Starting training for {num_epoch} epochs")
        
        self.models.train()
        
        for epoch in range(num_epoch):
            logger.info(f"Epoch {epoch + 1}/{num_epoch}")
            
            # Create data loaders
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
            
            batch_per_epoch_ = batch_per_epoch or min(len(dl) for dl in [DataLoader(ts, batch_size=self.batch_size) for ts in self.train_sets])
            
            metrics = []
            start_id = 0
            gradient_interval = min(batch_per_epoch_ - start_id, self.gradient_interval)
            
            # Training loop
            progress_bar = tqdm(range(batch_per_epoch_), desc=f"Epoch {epoch + 1}")
            
            for batch_id in progress_bar:
                batches = []
                
                # Get batch from each task
                for task_id, dataloader in enumerate(dataloaders):
                    try:
                        batch = next(dataloader)
                    except StopIteration:
                        # Restart dataloader if exhausted
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
                    
                    # Move batch to device
                    batch = self.move_to_device(batch)
                    batches.append(batch)
                
                # Forward pass
                loss, metric = self.models(batches)
                loss = loss / gradient_interval
                
                # Weight losses (first task has weight 1, others have weight tradeoff)
                weights = [1.0 if i == 0 else tradeoff for i in range(len(dataloaders))]
                all_loss = (loss * torch.tensor(weights, device=self.device)).sum()
                
                if not all_loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Check your model and loss computation.")
                
                all_loss.backward()
                metrics.append(metric)
                
                # Update parameters
                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Log metrics
                    if len(metrics) > 0:
                        avg_metrics = self.average_metrics(metrics)
                        progress_bar.set_postfix(avg_metrics)
                    
                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch_ - start_id, self.gradient_interval)
                    self.step += 1
            
            self.epoch += 1
            
            if self.scheduler:
                self.scheduler.step()

    def average_metrics(self, metrics_list):
        """Average metrics across batches"""
        if not metrics_list:
            return {}
        
        avg_metrics = defaultdict(float)
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    avg_metrics[key] += value
        
        for key in avg_metrics:
            avg_metrics[key] /= len(metrics_list)
        
        return dict(avg_metrics)

    def move_to_device(self, batch):
        """Move batch to device"""
        if isinstance(batch, dict):
            moved_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    moved_batch[key] = value.to(self.device)
                elif isinstance(value, dict):
                    moved_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                      for k, v in value.items()}
                else:
                    moved_batch[key] = value
            return moved_batch
        else:
            return batch

    @torch.no_grad()
    def evaluate(self, split, log=True):
        """
        Evaluate the model on specified split.
        """
        logger.info(f"Evaluating on {split} set")
        
        test_sets = getattr(self, f"{split}_sets")
        self.models.eval()
        
        all_metrics = defaultdict(float)
        
        for task_id, (test_set, model) in enumerate(zip(test_sets, self.models)):
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
                
                # Forward pass
                outputs = model(batch)
                
                # Compute metrics
                if hasattr(model, 'evaluate'):
                    metrics = model.evaluate(outputs, batch)
                else:
                    metrics = self.models.compute_default_metrics(outputs, batch)
                
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        task_metrics[key].append(value)
            
            # Average metrics for this task
            for key, values in task_metrics.items():
                if values:
                    avg_value = sum(values) / len(values)
                    metric_name = f"Task_{task_id} {key}"
                    if task_id == 0:
                        metric_name = "Center - " + metric_name
                    all_metrics[metric_name] = avg_value
        
        if log:
            logger.info("Evaluation Results:")
            for key, value in all_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        return dict(all_metrics)

    def save(self, checkpoint_path):
        """Save model checkpoint"""
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        state = {
            "epoch": self.epoch,
            "step": self.step,
            "optimizer": self.optimizer.state_dict(),
        }
        
        for i, model in enumerate(self.models):
            state[f"model_{i}"] = model.state_dict()
        
        torch.save(state, checkpoint_path)

    def load(self, checkpoint_path, load_optimizer=True):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        for i, model in enumerate(self.models):
            if f"model_{i}" in checkpoint:
                model.load_state_dict(checkpoint[f"model_{i}"])
        
        # Load optimizer state
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Load training state
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)


# Contrastive Loss Implementation
class ContrastiveLoss(nn.Module):
    """Contrastive loss for protein representation learning"""
    
    def __init__(self, temperature=0.1, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, embeddings1, embeddings2, labels=None):
        """
        Compute contrastive loss between two sets of embeddings
        
        Args:
            embeddings1: First set of embeddings [batch_size, embed_dim]
            embeddings2: Second set of embeddings [batch_size, embed_dim]
            labels: Optional labels for supervised contrastive learning
        """
        if self.normalize:
            embeddings1 = F.normalize(embeddings1, dim=1)
            embeddings2 = F.normalize(embeddings2, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        
        batch_size = embeddings1.size(0)
        
        if labels is None:
            # Self-supervised contrastive learning
            # Positive pairs are (i, i), negative pairs are (i, j) where i != j
            targets = torch.arange(batch_size, device=embeddings1.device)
            loss = F.cross_entropy(logits, targets)
        else:
            # Supervised contrastive learning
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float()
            
            # Mask out diagonal (self-similarity)
            mask = mask - torch.eye(batch_size, device=embeddings1.device)
            
            # Compute loss
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = -mean_log_prob_pos.mean()
        
        return loss


class MultiTaskWithContrastive(MultiTaskEngine):
    """Multi-task engine with contrastive learning"""
    
    def __init__(self, *args, contrastive_weight=0.1, temperature=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
    
    def compute_contrastive_loss(self, batches):
        """Compute contrastive loss between different task representations"""
        if len(batches) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Get embeddings from different tasks
        embeddings = []
        for i, (batch, model) in enumerate(zip(batches, self.models)):
            outputs = model(batch)
            graph_feature = outputs.get("graph_feature")
            if graph_feature is not None:
                embeddings.append(graph_feature)
        
        if len(embeddings) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute contrastive loss between first task and others
        total_contrastive_loss = 0
        for i in range(1, len(embeddings)):
            loss = self.contrastive_loss(embeddings[0], embeddings[i])
            total_contrastive_loss += loss
        
        return total_contrastive_loss / (len(embeddings) - 1)