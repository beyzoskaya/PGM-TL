from collections import defaultdict
import os
import sys
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
from protbert_hf import ProtBert
from collections import OrderedDict

logger = logging.getLogger(__name__)

"""
engine_hf.MultiTaskEngine.train creates PyTorch DataLoaders, 
uses collate_fn to batch items, and calls models in ModelsWrapper.forward
"""


TASK_TYPE_MAP = {
    "Task_0": "token_classification",    # SSP-Q8, 8 labels
    "Task_1": "regression",              # Stability, 1 label (numeric)
    "Task_2": "classification"           # HLA-MHC, 2 labels (binary classification)
}

class ModelsWrapper(nn.Module):

    def __init__(self, models, names):
        super(ModelsWrapper, self).__init__()
        self.models = nn.ModuleList(models)
        #print("Model names:", names)
        self.names = names

    def forward(self, batches):
        all_loss = []
        all_metric = defaultdict(float)
        
        for id, batch in enumerate(batches):
            model = self.models[id]
            task_type = getattr(model, 'task_type', None)
            
            if hasattr(model, 'compute_loss'):
                loss, metric = model.compute_loss(batch)
                print(f"Task {id} loss from model's compute_loss: {loss.item()}")
            else:
                outputs = model(batch)
                #print(f"[DEBUG] Shape of logits from Task {id}:", outputs["logits"].shape)
                loss = self.compute_default_loss(outputs, batch, task_type=task_type)
                #print(f"Task {id} loss from default compute_default_loss: {loss.item()}")
                metric = self.compute_default_metrics(outputs, batch)
            
            for k, v in metric.items():
                name = self.names[id] + " " + k
                if id == 0:
                    name = "Center - " + name
                all_metric[name] = v
            all_loss.append(loss)
        
        all_loss = torch.stack(all_loss)
        return all_loss, all_metric
    
    """
    SSP-Q8 --> token level --> CrossEntropy with attention mask (multi-class classification) / target shape: [batch, seq_len]
    Stability prediction --> sequence level --> MSE (regression) / target shape: [batch]
    HLA-MHC affinity --> classification (binary/multi-class) --> BCE or CrossEntropy / target shape: [batch]
    """

    def compute_default_loss(self, outputs, batch, task_type=None):
        logits = outputs["logits"]  # raw outputs of the final layer before softmax/sigmoid

        # Extract targets
        if isinstance(batch, dict) and 'targets' in batch:
            targets = batch['targets']
            if isinstance(targets, dict):
                #print("Targets is a dict, using the first key for loss computation.")
                #print("Available keys:", list(targets.keys()))
                target_key = list(targets.keys())[0]
                target = targets[target_key]  # pick first if multiple
                #print("Using target key:", target_key)
                #print("Targets example:", target if isinstance(target, (int, float, list)) else str(target)[:100])
            else:
                #print("Targets is not a dict, using it directly for loss computation.")
                #print("Targets type:", type(targets))
                #print("Targets example:", targets if isinstance(targets, (int, float, list)) else str(targets)[:100])
                target = targets
        else:
            raise ValueError("Cannot find targets in batch")

        #print("\n[DEBUG] ====== New Batch ======")
        #if "sequence" in batch:
        #    print("Example sequence:", batch["sequence"][0][:50], "...")
        #print("Targets (first item):", target[0] if isinstance(target, (list, torch.Tensor)) else target)
        #print("Logits shape:", logits.shape)

        # Determine task type
        task_type = task_type or getattr(self, 'task_type', None) or batch.get('task_type', 'regression')
        # Special case: force Task_2 to multi-class classification
        #if hasattr(self, 'names') and "Task_2" in self.names:
        #    task_type = "multi_class"
        #print("[DEBUG] task_type detected:", task_type)

        # --- Token-level classification (sequence labeling) ---
        if (isinstance(target, list) and len(target) > 0 and isinstance(target[0], list)) or \
        (isinstance(target, torch.Tensor) and target.dim() == 2):

            if isinstance(target, list):
                tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
                target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(logits.device)
            else:
                target_tensor = target.to(logits.device)

            mask = outputs.get("attention_mask")
            if mask is None:
                print("Warning: attention_mask not found, assuming all tokens are valid.")
                mask = torch.ones((target_tensor.size(0), logits.size(1)), dtype=torch.long, device=logits.device)
            else:
                mask = mask.to(logits.device)
            
            #print("[DEBUG] mask shape:", mask.shape)
            #print("[DEBUG] target_tensor shape after padding:", target_tensor.shape)

            if mask.size(1) != target_tensor.size(1):
                print(f"[WARNING] Sequence length mismatch: mask={mask.size(1)}, target={target_tensor.size(1)}")

            L_mask = mask.size(1)
            if target_tensor.size(1) > L_mask:
                target_tensor = target_tensor[:, :L_mask]
            elif target_tensor.size(1) < L_mask:
                pad = torch.full((target_tensor.size(0), L_mask - target_tensor.size(1)),
                                fill_value=-100, device=logits.device, dtype=target_tensor.dtype)
                target_tensor = torch.cat([target_tensor, pad], dim=1)

            active = mask.reshape(-1) == 1
            active_logits = logits.reshape(-1, logits.size(-1))[active]
            active_labels = target_tensor.reshape(-1)[active]

            if active_logits.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            return F.cross_entropy(active_logits, active_labels)  # token-level classification loss

        # --- Sequence-level tasks ---
        t = torch.tensor(target, device=logits.device, dtype=torch.float)

        if task_type == 'binary_classification':
            logits_ = logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            return F.binary_cross_entropy_with_logits(logits_, t.float())

        elif task_type == 'regression':
            if logits.numel() == t.numel():
                logits_flat = logits.view_as(t)
            elif logits.size(-1) == 1:
                logits_flat = logits.squeeze(-1)
            else:
                logits_flat = logits.mean(dim=1)
            return F.mse_loss(logits_flat, t.float())

        else:  # multi-class
            if logits.dim() == 1 or logits.size(-1) == 1:
                if logits.numel() == t.numel():
                    logits_flat = logits.view_as(t)
                else:
                    logits_flat = logits.squeeze(-1)
                return F.mse_loss(logits_flat, t.float())
            else:
                if t.dtype != torch.long:
                    t = t.long()
                t = torch.clamp(t, 0, logits.size(-1)-1)
                return F.cross_entropy(logits, t)

    def compute_default_metrics(self, outputs, batch, task_type=None):
        logits = outputs["logits"]
        #print("[DEBUG] logits shape:", logits.shape)
        #print("[DEBUG] batch keys:", batch.keys())

        # Extract targets
        if isinstance(batch, dict) and 'targets' in batch:
            targets = batch['targets']
            if isinstance(targets, dict):
                #print("Targets is a dict, using the first key for metric computation.")
                target_key = list(targets.keys())[0]
                target = targets[target_key]
            else:
                #print("Targets is not a dict, using it directly for metric computation.")
                target = targets
        else:
            raise ValueError("Cannot find targets in batch")

        #print("[DEBUG] raw targets type:", type(target))

        # Determine task type
        task_type = task_type or getattr(self, 'task_type', None) or batch.get('task_type', 'regression')
        # Force Task_2 as multi-class
        #if hasattr(self, 'names') and "Task_2" in self.names:
        #    task_type = "multi_class"
        #print("[DEBUG] task_type detected:", task_type)

        # --- Token-level classification ---
        if (isinstance(target, list) and len(target) > 0 and isinstance(target[0], list)) or \
        (isinstance(target, torch.Tensor) and target.dim() == 2):

            if isinstance(target, list):
                tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
                target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(logits.device)
            else:
                target_tensor = target.to(logits.device)

            mask = outputs.get("attention_mask")
            if mask is None:
                raise ValueError("Token-level task requires attention_mask")
            mask = mask.to(logits.device)

            L_mask = mask.size(1)
            if target_tensor.size(1) > L_mask:
                target_tensor = target_tensor[:, :L_mask]
            elif target_tensor.size(1) < L_mask:
                pad = torch.full((target_tensor.size(0), L_mask - target_tensor.size(1)),
                                fill_value=-100, device=logits.device, dtype=target_tensor.dtype)
                target_tensor = torch.cat([target_tensor, pad], dim=1)

            pred = logits.argmax(dim=-1)
            active = mask.reshape(-1) == 1
            active_preds = pred.reshape(-1)[active]
            active_labels = target_tensor.reshape(-1)[active]

            acc = (active_preds == active_labels).float().mean().item() if active_preds.numel() > 0 else 0.0
            #print("[DEBUG] Token-level task")
            #print("[DEBUG] target_tensor shape:", target_tensor.shape)
            #print("[DEBUG] mask shape:", mask.shape)
            #print("[DEBUG] pred shape:", pred.shape)
            #print("[DEBUG] active_preds size:", active_preds.size())
            #print("[DEBUG] active_labels size:", active_labels.size())
            return {"accuracy": acc}

        # --- Sequence-level tasks ---
        t = torch.tensor(target, device=logits.device, dtype=torch.float)

        if task_type == 'binary_classification':
            logits_ = logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            pred = (torch.sigmoid(logits_) > 0.5).long()
            #print("[DEBUG] Binary classification")
            return {"accuracy": (pred == t.long()).float().mean().item()}

        elif task_type == 'regression':
            if logits.numel() == t.numel():
                logits_flat = logits.view_as(t)
            elif logits.size(-1) == 1:
                logits_flat = logits.squeeze(-1)
            else:
                logits_flat = logits.mean(dim=1)
            mse = F.mse_loss(logits_flat, t.float()).item()
            #print("[DEBUG] Regression")
            return {"mse": mse}

        else:  # multi-class classification
            if t.dtype != torch.long:
                t = t.long()
            t = torch.clamp(t, 0, logits.size(-1)-1)
            pred = logits.argmax(dim=-1)
            acc = (pred == t).float().mean().item()
            #print("[DEBUG] Multi-class classification")
            return {"accuracy": acc}

    def __getitem__(self, id):
        return self.models[id]

class MultiTaskEngine:

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

    # collate keeps token-level labels as lists so that padding can be handled in compute_loss
    # collate converts sequence-level labels into tensors assuming these are numeric
    # collate does not tokenize or pad sequences - returns a list of raw sequences (tokenization and padding occur inside the backbone model)
    def collate_fn(self, batch):

        sequences = []
        targets_list = []
        
        for item in batch:
            if isinstance(item, dict):
                sequences.append(item['sequence'])
                targets_list.append(item.get('targets', {}))
            else:
                sequences.append(getattr(item, 'sequence', ''))
                targets_list.append({})

        combined_targets = defaultdict(list)
        for targets in targets_list:
            if isinstance(targets, dict):
                for key, value in targets.items():
                    combined_targets[key].append(value)
        
        # TOKEN-LEVEL vs SEQUENCE-LEVEL
        target_tensors = {}
        for key, values in combined_targets.items():
            # if this is token-level data (list of lists)
            if values and isinstance(values[0], list):
                # Token-level: nested list
                target_tensors[key] = values  
            else:
                # Sequence-level: tensor
                try:
                    target_tensors[key] = torch.tensor(values, dtype=torch.float)
                except:
                    target_tensors[key] = values
        
        return {
            'sequence': sequences,
            'targets': target_tensors
        }

    def train(self, num_epoch=1, batch_per_epoch=None, tradeoff=1.0):
      
        logger.info(f"Starting training for {num_epoch} epochs")
        
        self.models.train()
        
        for epoch in range(num_epoch):
            logger.info(f"Epoch {epoch + 1}/{num_epoch}")
            
            # creates one DataLoader per task and store iterator objects
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
            
            progress_bar = tqdm(range(batch_per_epoch_), desc=f"Epoch {epoch + 1}")
            
            for batch_id in progress_bar:
                batches = []
                
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
                
                loss, metric = self.models(batches)
                loss = loss / gradient_interval
                
                # Weight losses (first task has weight 1, others have weight tradeoff)
                # loss is a per-task vector; dividing by gradient_interval to average over steps
                # center task strategy - first task is the main task with full weight
                # Task 1,2 auxiliary tasks with reduced weight

                # FIXME: Without weighting, one task could dominate training!
                weights = [1.0 if i == 0 else tradeoff for i in range(len(dataloaders))]
                all_loss = (loss * torch.tensor(weights, device=self.device)).sum()
                
                if not all_loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Check your model and loss computation.")
                
                all_loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
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
            total_samples = 0
            total_correct = 0  # For accuracy debugging

            for batch in tqdm(dataloader, desc=f"Evaluating Task {task_id}"):
                batch = self.move_to_device(batch)

                # Forward pass
                outputs = model(batch)
                #print(f"[DEBUG] Eval Task {task_id} logits shape:", outputs["logits"].shape)

                if hasattr(model, 'evaluate'):
                    metrics = model.evaluate(outputs, batch)
                else:
                    task_type = getattr(model, 'task_type', None)
                    metrics = self.models.compute_default_metrics(outputs, batch, task_type=task_type)

                if 'labels' in batch:
                    batch_size = batch['labels'].shape[0] if isinstance(batch['labels'], torch.Tensor) else len(batch['labels'])
                elif 'targets' in batch:
                    batch_size = batch['targets'].shape[0] if isinstance(batch['targets'], torch.Tensor) else len(batch['targets'])
                else:
                    batch_size = 1  # fallback

                total_samples += batch_size

                if 'accuracy' in metrics:
                    total_correct += metrics['accuracy'] * batch_size

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

            # Log task-level debug info
            computed_accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0
            logger.info(f"Task {task_id}: Total samples = {total_samples}, Computed accuracy = {computed_accuracy:.4f}")

        if log:
            logger.info("Evaluation Results:")
            for key, value in all_metrics.items():
                logger.info(f"  {key}: {value:.4f}")

        return dict(all_metrics)

    def save(self, checkpoint_path):
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


class SharedBackboneMultiTaskModel(nn.Module):

    def __init__(self, model_name="Rostlab/prot_bert_bfd", tasks_config=None, 
                 readout="pooler", freeze_bert=False, lora_config=None):
        super().__init__()
        
        self.shared_backbone = ProtBert(
            model_name=model_name, 
            readout=readout, 
            freeze_bert=freeze_bert
        )
        
        if lora_config:
            self.add_lora_adapters(**lora_config)
        
        self.task_heads = nn.ModuleDict()
        self.task_types = {}
        
        if tasks_config:
            for task_id, task_config in enumerate(tasks_config):
                task_name = f"task_{task_id}"
                self.task_types[task_name] = task_config['type']
                
                head = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(self.shared_backbone.output_dim, task_config['num_labels'])
                )
                self.task_heads[task_name] = head

        # Initialize weights
        for head in self.task_heads.values():
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def add_lora_adapters(self, rank=16, alpha=32, dropout=0.1):
        from protbert_hf import LoRALinear
        for layer in self.shared_backbone.model.encoder.layer:
            attention = layer.attention.self
            for name in ['query', 'key', 'value']:
                if hasattr(attention, name):
                    original_layer = getattr(attention, name)
                    lora_layer = LoRALinear(
                        original_layer.in_features,
                        original_layer.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout
                    )
                    lora_layer.linear.weight.data = original_layer.weight.data.clone()
                    setattr(attention, name, lora_layer)
    
    def forward(self, batch, task_id):
        """
        batch: dict with input features + 'labels'
        task_id: int
        returns: dict with logits and features
        """
        task_name = f"task_{task_id}"
        task_type = self.task_types[task_name]
        head = self.task_heads[task_name]

        # Shared backbone forward
        backbone_outputs = self.shared_backbone(batch)

        # Select correct features
        if task_type == 'token_classification':
            features = backbone_outputs["residue_feature"]  # [B, L, H]
        else:
            features = backbone_outputs["graph_feature"]  # [B, H]

        logits = head(features)

        # Optional: flatten logits for token classification
        if task_type == 'token_classification':
            # CrossEntropyLoss expects [B*L, C] vs [B*L]
            logits_for_loss = logits.view(-1, logits.size(-1))
            labels_for_loss = batch['labels'].view(-1)
        else:
            logits_for_loss = logits
            labels_for_loss = batch['labels']

        return {
            "logits": logits,
            "logits_for_loss": logits_for_loss,
            "labels_for_loss": labels_for_loss,
            "graph_feature": backbone_outputs["graph_feature"],
            "residue_feature": backbone_outputs["residue_feature"],
            "attention_mask": backbone_outputs.get("attention_mask")
        }

    def get_task_model(self, task_id):
        return TaskModelWrapper(self, task_id)



class TaskModelWrapper(nn.Module):
    
    def __init__(self, shared_model, task_id):
        super().__init__()
        self.shared_model = shared_model
        self.task_id = task_id
        self.task_type = shared_model.task_types[f"task_{task_id}"]
    
    def forward(self, batch):
        return self.shared_model(batch, self.task_id)
    
    def parameters(self):
        return self.shared_model.parameters()


class SharedBackboneModelsWrapper(nn.Module):
  
    def __init__(self, shared_model, task_names):
        super().__init__()
        self.shared_model = shared_model
        self.names = task_names
        
        self.task_wrappers = nn.ModuleList([
            shared_model.get_task_model(i) for i in range(len(task_names))
        ])
    
    def forward(self, batches):
        all_loss = []
        all_metric = {}
        
        for task_id, batch in enumerate(batches):
            task_wrapper = self.task_wrappers[task_id]
            task_type = getattr(task_wrapper, 'task_type', None)
            
            # Forward pass through shared model for this task
            outputs = task_wrapper(batch)
            
            # Compute loss (reuse existing loss computation)
            loss = self.compute_default_loss(outputs, batch, task_type=task_type)
            metric = self.compute_default_metrics(outputs, batch, task_type=task_type)
            
            # Store metrics with task names
            for k, v in metric.items():
                name = self.names[task_id] + " " + k
                if task_id == 0:
                    name = "Center - " + name
                all_metric[name] = v
            
            all_loss.append(loss)
        
        all_loss = torch.stack(all_loss)
        return all_loss, all_metric
    
    def compute_default_loss(self, outputs, batch, task_type=None):
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
            raise ValueError("Cannot find targets in batch")

        # Determine task type
        task_type = task_type or getattr(self, 'task_type', None) or batch.get('task_type', 'regression')

        # --- Token-level classification (sequence labeling) ---
        if (isinstance(target, list) and len(target) > 0 and isinstance(target[0], list)) or \
        (isinstance(target, torch.Tensor) and target.dim() == 2):

            if isinstance(target, list):
                tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
                target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(logits.device)
            else:
                target_tensor = target.to(logits.device)

            mask = outputs.get("attention_mask")
            if mask is None:
                print("Warning: attention_mask not found, assuming all tokens are valid.")
                mask = torch.ones((target_tensor.size(0), logits.size(1)), dtype=torch.long, device=logits.device)
            else:
                mask = mask.to(logits.device)

            L_mask = mask.size(1)
            if target_tensor.size(1) > L_mask:
                target_tensor = target_tensor[:, :L_mask]
            elif target_tensor.size(1) < L_mask:
                pad = torch.full((target_tensor.size(0), L_mask - target_tensor.size(1)),
                                fill_value=-100, device=logits.device, dtype=target_tensor.dtype)
                target_tensor = torch.cat([target_tensor, pad], dim=1)

            active = mask.reshape(-1) == 1
            active_logits = logits.reshape(-1, logits.size(-1))[active]
            active_labels = target_tensor.reshape(-1)[active]

            if active_logits.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            return F.cross_entropy(active_logits, active_labels)

        # --- Sequence-level tasks ---
        if not torch.is_tensor(target):
            t = torch.tensor(target, device=logits.device, dtype=torch.float)
        else:
            t = target.to(device=logits.device, dtype=torch.float)

        if task_type == 'binary_classification':
            # Ensure logits are properly shaped for binary classification
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits_flat = logits.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            else:
                logits_flat = logits.view(-1)     # Flatten to [batch_size]
            
            # Ensure targets are properly shaped
            t_flat = t.view(-1)  # Flatten to [batch_size]
            
            # Use standard BCE loss without pos_weight to avoid complications
            return F.binary_cross_entropy_with_logits(logits_flat, t_flat)

        elif task_type == 'regression':
            if logits.numel() == t.numel():
                logits_flat = logits.view_as(t)
            elif logits.size(-1) == 1:
                logits_flat = logits.squeeze(-1)
            else:
                logits_flat = logits.mean(dim=1)
            return F.mse_loss(logits_flat, t.float())

        else:  # multi-class classification
            if logits.dim() == 1 or logits.size(-1) == 1:
                if logits.numel() == t.numel():
                    logits_flat = logits.view_as(t)
                else:
                    logits_flat = logits.squeeze(-1)
                return F.mse_loss(logits_flat, t.float())
            else:
                if t.dtype != torch.long:
                    t = t.long()
                t = torch.clamp(t, 0, logits.size(-1)-1)
                return F.cross_entropy(logits, t)

    def compute_default_metrics(self, outputs, batch, task_type=None):
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
            raise ValueError("Cannot find targets in batch")

        # Determine task type
        task_type = task_type or getattr(self, 'task_type', None) or batch.get('task_type', 'regression')

        # --- Token-level classification ---
        if (isinstance(target, list) and len(target) > 0 and isinstance(target[0], list)) or \
        (isinstance(target, torch.Tensor) and target.dim() == 2):

            if isinstance(target, list):
                tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
                target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(logits.device)
            else:
                target_tensor = target.to(logits.device)

            mask = outputs.get("attention_mask")
            if mask is None:
                raise ValueError("Token-level task requires attention_mask")
            mask = mask.to(logits.device)

            L_mask = mask.size(1)
            if target_tensor.size(1) > L_mask:
                target_tensor = target_tensor[:, :L_mask]
            elif target_tensor.size(1) < L_mask:
                pad = torch.full((target_tensor.size(0), L_mask - target_tensor.size(1)),
                                fill_value=-100, device=logits.device, dtype=target_tensor.dtype)
                target_tensor = torch.cat([target_tensor, pad], dim=1)

            pred = logits.argmax(dim=-1)
            active = mask.reshape(-1) == 1
            active_preds = pred.reshape(-1)[active]
            active_labels = target_tensor.reshape(-1)[active]

            acc = (active_preds == active_labels).float().mean().item() if active_preds.numel() > 0 else 0.0
            return {"accuracy": acc}

        # --- Sequence-level tasks ---
        t = torch.tensor(target, device=logits.device, dtype=torch.float)

        if task_type == 'binary_classification':
            # Properly handle binary classification logits
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits_flat = logits.squeeze(-1)
            else:
                logits_flat = logits.view(-1)
            
            # Apply sigmoid and threshold at 0.5
            pred = (torch.sigmoid(logits_flat) > 0.5).long()
            t_long = t.long()
            
            return {"accuracy": (pred == t_long).float().mean().item()}

        elif task_type == 'regression':
            if logits.numel() == t.numel():
                logits_flat = logits.view_as(t)
            elif logits.size(-1) == 1:
                logits_flat = logits.squeeze(-1)
            else:
                logits_flat = logits.mean(dim=1)
            mse = F.mse_loss(logits_flat, t.float()).item()
            return {"mse": mse}

        else:  # multi-class classification
            if t.dtype != torch.long:
                t = t.long()
            t = torch.clamp(t, 0, logits.size(-1)-1)
            pred = logits.argmax(dim=-1)
            acc = (pred == t).float().mean().item()
            return {"accuracy": acc}
    
    def __getitem__(self, idx):
        return self.task_wrappers[idx]


def create_shared_multitask_model(tasks_config, model_config):
 
    readout = 'pooler'
    for task_cfg in tasks_config:
        if task_cfg['type'] == 'token_classification':
            readout = 'per_token'
            break

    readout = model_config.get('readout', readout)

    # LoRA config if applicable
    lora_config = None
    if model_config.get('type') in ['lora', 'shared_lora']:
        lora_config = {
            'rank': model_config.get('lora_rank', 16),
            'alpha': model_config.get('lora_alpha', 32),
            'dropout': model_config.get('lora_dropout', 0.1)
        }

    shared_model = SharedBackboneMultiTaskModel(
        model_name=model_config['model_name'],
        tasks_config=tasks_config,
        readout=readout,
        freeze_bert=model_config.get('freeze_bert', False),
        lora_config=lora_config
    )

    return shared_model
