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

    def __init__(self, models, names):
        super(ModelsWrapper, self).__init__()
        self.models = nn.ModuleList(models)
        print("Model names:", names)
        self.names = names

    def forward(self, batches):
        all_loss = []
        all_metric = defaultdict(float)
        
        for id, batch in enumerate(batches):
            if hasattr(self.models[id], 'compute_loss'):
                loss, metric = self.models[id].compute_loss(batch)
                print(f"Task {id} loss from model's compute_loss: {loss.item()}")
            else:
                outputs = self.models[id](batch)
                loss = self.compute_default_loss(outputs, batch)
                print(f"Task {id} loss from default compute_default_loss: {loss.item()}")
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
        """
        Compute default loss supporting:
        - sequence-level classification/regression
        - token-level classification (list-of-lists or 2D tensor)
        Uses outputs['attention_mask'] (residue-level mask) to ignore padding.
        """
        logits = outputs["logits"]

        # --- Extract targets from batch ---
        if isinstance(batch, dict) and 'targets' in batch:
            targets = batch['targets']
            if isinstance(targets, dict):
                target_key = list(targets.keys())[0]
                target = targets[target_key]
            else:
                target = targets
        else:
            raise ValueError("Cannot find targets in batch")

        # Helper: return a zero loss that requires grad (used when no active positions)
        def zero_loss():
            z = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return z

        # --- Case A: target is list ---
        if isinstance(target, list):
            # Token-level: list of lists (per-residue labels)
            if len(target) > 0 and isinstance(target[0], list):
                from torch.nn.utils.rnn import pad_sequence
                tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
                target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(logits.device)

                mask = outputs.get("attention_mask")
                if mask is None:
                    # Fallback: assume all positions valid up to logits length
                    mask = torch.ones((len(target), logits.size(1)), dtype=torch.long, device=logits.device)
                else:
                    mask = mask.to(logits.device)

                # Align target length with mask/logits length
                L_mask = mask.size(1)
                if target_tensor.size(1) > L_mask:
                    target_tensor = target_tensor[:, :L_mask]
                elif target_tensor.size(1) < L_mask:
                    pad = torch.full((target_tensor.size(0), L_mask - target_tensor.size(1)),
                                    fill_value=-100, device=logits.device, dtype=target_tensor.dtype)
                    target_tensor = torch.cat([target_tensor, pad], dim=1)

                # Flatten/select active positions using reshape (works for non-contiguous)
                active = mask.reshape(-1) == 1
                active_logits = logits.reshape(-1, logits.size(-1))[active]
                active_labels = target_tensor.reshape(-1)[active]

                if active_logits.numel() == 0:
                    return zero_loss()

                loss = F.cross_entropy(active_logits, active_labels)
                return loss

            # Sequence-level: list of scalars
            else:
                if logits.dim() == 1 or logits.size(-1) == 1:
                    t = torch.tensor(target, dtype=torch.float, device=logits.device)
                    loss = F.mse_loss(logits.squeeze(), t)
                else:
                    t = torch.tensor(target, dtype=torch.long, device=logits.device)
                    max_class = logits.size(-1) - 1
                    t = torch.clamp(t, 0, max_class)
                    loss = F.cross_entropy(logits, t)
                return loss

        # --- Case B: target is a tensor ---
        if isinstance(target, torch.Tensor):
            target = target.to(logits.device)

            # Token-level as 2D tensor [batch, seq_len]
            if target.dim() == 2:
                mask = outputs.get("attention_mask")
                if mask is None:
                    # If missing, assume all positions valid up to logits length
                    mask = torch.ones((target.size(0), logits.size(1)), dtype=torch.long, device=logits.device)
                else:
                    mask = mask.to(logits.device)

                # Align lengths
                L_mask = mask.size(1)
                if target.size(1) > L_mask:
                    target = target[:, :L_mask]
                elif target.size(1) < L_mask:
                    pad = torch.full((target.size(0), L_mask - target.size(1)),
                                    fill_value=-100, device=logits.device, dtype=target.dtype)
                    target = torch.cat([target, pad], dim=1)

                active = mask.reshape(-1) == 1
                active_logits = logits.reshape(-1, logits.size(-1))[active]
                active_labels = target.reshape(-1)[active]

                if active_logits.numel() == 0:
                    return zero_loss()

                loss = F.cross_entropy(active_logits, active_labels)
                return loss

            # Sequence-level tensor (1D)
            else:
                if logits.dim() == 1 or logits.size(-1) == 1:
                    loss = F.mse_loss(logits.squeeze(), target.float())
                else:
                    if target.dtype != torch.long:
                        target = target.long()
                    max_class = logits.size(-1) - 1
                    target = torch.clamp(target, 0, max_class)
                    loss = F.cross_entropy(logits, target)
                return loss

        # --- Fallback ---
        target = torch.tensor(target, dtype=torch.float, device=logits.device)
        if logits.dim() == 1 or logits.size(-1) == 1:
            loss = F.mse_loss(logits.squeeze(), target.float())
        else:
            loss = F.cross_entropy(logits, target.long())

        return loss

    
    def compute_default_metrics(self, outputs, batch):
        """
        Compute default metrics for classification/regression.
        Token-level tasks use outputs['attention_mask'] to select valid residues.
        """
        logits = outputs["logits"]

        # --- Extract targets ---
        if isinstance(batch, dict) and 'targets' in batch:
            targets = batch['targets']
            if isinstance(targets, dict):
                target_key = list(targets.keys())[0]
                target = targets[target_key]
            else:
                target = targets
        else:
            raise ValueError("Cannot find targets in batch")

        # --- Token-level (list of lists or 2D tensor) ---
        if isinstance(target, list) or (isinstance(target, torch.Tensor) and target.dim() == 2):
            if isinstance(target, list):
                from torch.nn.utils.rnn import pad_sequence
                tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
                target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(logits.device)
            else:
                target_tensor = target.to(logits.device)

            mask = outputs.get("attention_mask")
            if mask is None:
                raise ValueError("Token-level task requires attention_mask from the model outputs")
            mask = mask.to(logits.device)

            # Align lengths with mask
            L_mask = mask.size(1)
            if target_tensor.size(1) > L_mask:
                target_tensor = target_tensor[:, :L_mask]
            elif target_tensor.size(1) < L_mask:
                pad = torch.full((target_tensor.size(0), L_mask - target_tensor.size(1)),
                                fill_value=-100, device=logits.device, dtype=target_tensor.dtype)
                target_tensor = torch.cat([target_tensor, pad], dim=1)

            # Predictions
            pred = logits.argmax(dim=-1)

            # Use reshape to handle non-contiguous tensors
            active = mask.reshape(-1) == 1
            active_preds = pred.reshape(-1)[active]
            active_labels = target_tensor.reshape(-1)[active]

            if active_preds.numel() == 0:
                return {"accuracy": 0.0}

            acc = (active_preds == active_labels).float().mean().item()
            return {"accuracy": acc}

        # --- Sequence-level classification/regression ---
        if isinstance(target, torch.Tensor) and target.dim() == 1:
            target = target.to(logits.device)
            if logits.dim() == 2 and logits.size(-1) > 1:
                pred = logits.argmax(dim=-1)
                acc = (pred == target.long()).float().mean().item()
                return {"accuracy": acc}
            else:
                mse = F.mse_loss(logits.squeeze(), target.float()).item()
                return {"mse": mse}

        # --- Fallback: list of scalars ---
        if isinstance(target, list):
            target_tensor = torch.tensor(target, device=logits.device)
            if logits.dim() == 2 and logits.size(-1) > 1:
                pred = logits.argmax(dim=-1)
                acc = (pred == target_tensor.long()).float().mean().item()
                return {"accuracy": acc}
            else:
                mse = F.mse_loss(logits.squeeze(), target_tensor.float()).item()
                return {"mse": mse}

        return {}



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
                sequences.append(getattr(item, 'sequence', ''))
                targets_list.append({})
        
        # Combine targets
        combined_targets = defaultdict(list)
        for targets in targets_list:
            if isinstance(targets, dict):
                for key, value in targets.items():
                    combined_targets[key].append(value)
        
        # Convert to tensors - HANDLE TOKEN-LEVEL vs SEQUENCE-LEVEL
        target_tensors = {}
        for key, values in combined_targets.items():
            # Check if this is token-level data (list of lists)
            if values and isinstance(values[0], list):
                # Token-level: keep as nested list, don't convert to tensor yet
                target_tensors[key] = values  # Keep as list of lists
            else:
                # Sequence-level: convert to tensor
                try:
                    target_tensors[key] = torch.tensor(values, dtype=torch.float)
                except:
                    target_tensors[key] = values
        
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