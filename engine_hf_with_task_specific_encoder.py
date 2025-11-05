from collections import defaultdict
import os
import sys
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import pickle

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class TaskModel(nn.Module):
    
    def __init__(
        self, 
        backbone, 
        task_type, 
        num_labels, 
        task_id, 
        hidden_dim=768, 
        use_task_encoder=True, 
        encoder_dim=512,
        use_bilstm_for_token=True,  
        lstm_hidden_dim=512,           
        lstm_layers=1,                
        lstm_dropout=0.3
    ):
        super().__init__()
        self.backbone = backbone
        self.task_type = task_type
        self.num_labels = num_labels
        self.task_id = task_id
        self.output_dim = backbone.output_dim
        self.use_task_encoder = use_task_encoder
        self.use_bilstm_for_token = use_bilstm_for_token and (task_type == "token_classification")

        # Task-specific encoder (shared structure for all tasks)
        if use_task_encoder:
            self.task_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.2)
            )
            self.encoder_norm = nn.LayerNorm(hidden_dim)
        else:
            self.task_encoder = None
            self.encoder_norm = None

        # Optional BiLSTM for token-level tasks (secondary structure)
        if self.use_bilstm_for_token:
            self.bilstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=lstm_dropout if lstm_layers > 1 else 0.0,
                bidirectional=True
            )
            bilstm_output_dim = lstm_hidden_dim * 2
            head_input_dim = bilstm_output_dim
        else:
            self.bilstm = None
            head_input_dim = hidden_dim

        # Final task-specific output head
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(head_input_dim, num_labels)
        )

    def forward(self, batch):
        """
        Forward pass with shape validation and alignment.
        
        Args:
            batch: {'sequence': [...], 'targets': {...}}
        
        Returns:
            {'logits': [...], 'attention_mask': [...]}
        """
        backbone_out = self.backbone(batch)

        # --- Get feature type based on task ---
        if self.task_type == 'token_classification':
            features = backbone_out["residue_feature"]  # [batch, seq_len, hidden]
        else:
            features = backbone_out["graph_feature"]    # [batch, hidden]

        # --- Validate and fix attention mask for token-level tasks ---
        if self.task_type == 'token_classification':
            attn_mask = backbone_out.get('attention_mask')
            if attn_mask is not None:
                # Check shape alignment
                if attn_mask.size(1) != features.size(1):
                    logger.warning(
                        f"[TaskModel {self.task_id}] Attention mask shape mismatch: "
                        f"mask {attn_mask.shape} vs features {features.shape}. Aligning..."
                    )
                    # Align to features shape
                    if attn_mask.size(1) > features.size(1):
                        attn_mask = attn_mask[:, :features.size(1)]
                    else:
                        pad_size = features.size(1) - attn_mask.size(1)
                        pad = torch.ones(
                            (attn_mask.size(0), pad_size),
                            device=attn_mask.device,
                            dtype=attn_mask.dtype
                        )
                        attn_mask = torch.cat([attn_mask, pad], dim=1)
                    # Update in backbone_out for downstream use
                    backbone_out['attention_mask'] = attn_mask

        # --- Task-specific encoder ---
        if self.use_task_encoder and self.task_encoder is not None:
            if self.task_type == 'token_classification':
                batch_size, seq_len, hidden_dim = features.shape
                features = features.reshape(-1, hidden_dim)
                features = self.task_encoder(features)
                features = self.encoder_norm(features)
                features = features.reshape(batch_size, seq_len, hidden_dim)
            else:
                features = self.task_encoder(features)
                features = self.encoder_norm(features)

        # --- BiLSTM for token-level tasks ---
        if self.use_bilstm_for_token and self.bilstm is not None:
            self.bilstm.flatten_parameters()
            features, _ = self.bilstm(features)  # [batch, seq_len, 2*lstm_hidden_dim]

        # --- Final linear head ---
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
        self.loss_running_mean = [1.0] * len(task_models)
    
    def forward(self, batches, normalize_losses=True):
        """
        Args:
            batches: list of task-specific batches

        Returns:
            all_loss: [num_tasks] - per-task losses
            all_metric: dict of per-task metrics
        """
        all_loss_raw = []
        all_metric = {}

        for task_id, batch in enumerate(batches):
            model = self.task_models[task_id]
            task_type = model.task_type

            outputs = model(batch)
            logits = outputs['logits']

            # --- Compute raw loss (no normalization) ---
            loss_raw = self._compute_loss(logits, batch, task_type)
            all_loss_raw.append(loss_raw)

            # --- Compute metrics ---
            metric = self._compute_metrics(logits, batch, task_type)
            for k, v in metric.items():
                metric_name = f"{self.task_names[task_id]} {k}"
                all_metric[metric_name] = v

        all_loss_raw = torch.stack(all_loss_raw)

        if normalize_losses:
            norm_losses = []
            for i, lr in enumerate(all_loss_raw):
                denom = self.loss_running_mean[i] + 1e-8
                norm_losses.append(lr / denom)
            all_loss_norm = torch.stack(norm_losses)
        else:
            all_loss_norm = all_loss_raw.clone()

        loss_list = [round(l.item(), 4) for l in all_loss_raw]
        norm_list = [round(l.item(), 4) for l in all_loss_norm]
        logger.debug(f"[Forward] Raw losses={loss_list}, Normalized={norm_list}")

        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 100 == 0:
            loss_vals = [round(l.item(), 5) for l in all_loss_raw]
            try:
                from tqdm import tqdm
                tqdm.write(f"[Debug] Step {self._debug_counter}: Raw per-task losses {loss_vals}")
            except Exception:
                print(f"[Debug] Step {self._debug_counter}: Raw per-task losses {loss_vals}")

        return all_loss_norm, all_metric, all_loss_raw

    def update_loss_statistics(self, losses_raw):
        for i, loss in enumerate(losses_raw):
            loss_val = loss.detach().item()
            self.loss_running_mean[i] = 0.9 * self.loss_running_mean[i] + 0.1 * loss_val
    
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
        """
        Token-level loss: cross-entropy with attention mask and class weighting.
        Handles sequence length misalignment and validates all inputs.
        """
        device = logits.device
        num_classes = logits.size(-1)
        
        # --- target to tensor ---
        if isinstance(target, list):
            tgt_tensors = [torch.tensor(t, dtype=torch.long, device=device) for t in target]
            target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100)
        else:
            target_tensor = target.to(device).long()
        
        # --- attention mask ---
        attention_mask = batch.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones(
                (logits.size(0), logits.size(1)), 
                dtype=torch.long, 
                device=device
            )
        else:
            attention_mask = attention_mask.to(device).long()
        
        seq_len = logits.size(1)
        batch_size = logits.size(0)
        
        # --- Align target_tensor to seq_len ---
        if target_tensor.size(1) > seq_len:
            target_tensor = target_tensor[:, :seq_len]
        elif target_tensor.size(1) < seq_len:
            pad_size = seq_len - target_tensor.size(1)
            pad = torch.full(
                (batch_size, pad_size),
                fill_value=-100,
                device=device,
                dtype=target_tensor.dtype
            )
            target_tensor = torch.cat([target_tensor, pad], dim=1)
        
        # --- Align attention_mask to seq_len ---
        if attention_mask.size(1) > seq_len:
            attention_mask = attention_mask[:, :seq_len]
        elif attention_mask.size(1) < seq_len:
            pad_size = seq_len - attention_mask.size(1)
            pad = torch.ones(
                (batch_size, pad_size),
                dtype=attention_mask.dtype,
                device=device
            )
            attention_mask = torch.cat([attention_mask, pad], dim=1)
        
        # --- Validate target labels are in valid range ---
        valid_labels = (target_tensor >= -1) & (target_tensor < num_classes)
        if not valid_labels.all():
            invalid_count = (~valid_labels).sum().item()
            logger.warning(
                f"Found {invalid_count} invalid labels. "
                f"Range: [{target_tensor.min()}, {target_tensor.max()}], "
                f"Expected: [0, {num_classes-1}]"
            )
            # Set invalid labels to -100 (ignored by cross_entropy)
            target_tensor[~valid_labels] = -100
        
        # --- Identify active tokens (not masked and not padding) ---
        active = attention_mask.reshape(-1) == 1
        active_count = active.sum().item()
        
        if active_count == 0:
            logger.warning(f"Batch has NO active tokens! Returning small loss.")
            return torch.tensor(1e-6, device=device, dtype=logits.dtype, requires_grad=True)
        
        # --- Extract active logits and labels ---
        active_logits = logits.reshape(-1, num_classes)[active]  # [active_count, num_classes]
        active_labels = target_tensor.reshape(-1)[active]        # [active_count]
        
        # --- Compute class weights for handling class imbalance (e.g., in secondary structure) ---
        # Only compute from valid (non-ignored) labels
        valid_label_mask = active_labels >= 0
        if valid_label_mask.sum() > 0:
            valid_labels_only = active_labels[valid_label_mask]
            class_counts = torch.bincount(valid_labels_only, minlength=num_classes).float()
            
            # Inverse frequency weighting
            class_weights = torch.where(
                class_counts > 0,
                1.0 / (class_counts + 1e-8),
                torch.ones(num_classes, device=device, dtype=torch.float32)
            )
            class_weights = class_weights / class_weights.sum()
        else:
            class_weights = torch.ones(num_classes, device=device, dtype=torch.float32) / num_classes
        
        # --- Compute loss with class weighting and ignore_index ---
        loss = F.cross_entropy(
            active_logits,
            active_labels,
            weight=class_weights,
            ignore_index=-100,
            reduction='mean'
        )
        
        # --- Validate loss value ---
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f"Loss is NaN/Inf! active_count={active_count}, "
                f"unique_labels={active_labels.unique().tolist()}, "
                f"loss={loss.item()}"
            )
            return torch.tensor(1e-6, device=device, dtype=logits.dtype, requires_grad=True)
        
        return loss
    
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

    """
    Task weights (size_norm) are computed as inversely proportional to the square root of dataset size. This balances tasks during training.
    Task models are wrapped in a ModelsWrapper that handles batch forwarding across tasks.
    Sequence-level tasks (single scalar outputs per sequence) are detected automatically.
    """
    
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets, 
                 optimizer, scheduler=None, batch_size=8, gradient_interval=1, 
                 num_worker=0, log_interval=100, device='cuda',
                 use_task_encoder=True, encoder_dim=512):
        
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker
        self.log_interval = log_interval
        
        # --- Compute more balanced task weights (1/sqrt(N)) ---
        self.task_dataset_sizes = [len(ts) for ts in train_sets]
        sizes = torch.tensor(self.task_dataset_sizes, dtype=torch.float)
        inv_sqrt = 1.0 / torch.sqrt(sizes + 1e-8)    # avoid divide-by-zero
        self.task_size_weights = (inv_sqrt / inv_sqrt.sum()).float()
        
        # --- Task models with optional task-specific encoders ---
        task_models = []
        for task_id, config in enumerate(task_configs):
            use_bilstm_for_token = (config["type"] == "token_classification")
            model = TaskModel(
                backbone=backbone,
                task_type=config['type'],
                num_labels=config['num_labels'],
                task_id=task_id,
                use_task_encoder=use_task_encoder,
                hidden_dim=backbone.output_dim,
                encoder_dim=encoder_dim,
                use_bilstm_for_token=use_bilstm_for_token
            )
            task_models.append(model)
            logger.info(
                f"[Init] Task {task_id} ({config['type']}): "
                f"{'Using BiLSTM head for token classification' if use_bilstm_for_token else 'Using standard dense head'}"
            )
        
        task_names = [f"Task_{i}" for i in range(len(task_configs))]
        self.models = ModelsWrapper(task_models, task_names)
        self.models.to(self.device)

        # --- Detect sequence-level tasks automatically ---
        self.sequence_level_tasks = [
            name for name, cfg in zip(task_names, task_configs)
            if cfg['type'] in ('binary_classification', 'regression')
        ]
        
        self.train_sets = train_sets
        self.valid_sets = valid_sets
        self.test_sets = test_sets
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.log_sigma = nn.Parameter(torch.zeros(len(task_configs), device=self.device))
        sigma_in_optimizer = any(
            self.log_sigma in g['params'] for g in self.optimizer.param_groups
        )

        if not sigma_in_optimizer:
            self.optimizer.add_param_group({
                'params': [self.log_sigma],
                'lr': self.optimizer.param_groups[0]['lr'],
                'weight_decay': 0.0  # Don't regularize uncertainties
            })
            logger.info(f"✓ Added log_sigma to optimizer param groups")
        else:
            logger.info(f"✓ log_sigma already in optimizer")

        logger.info(f"Optimizer now has {len(self.optimizer.param_groups)} param groups")
        logger.info(f"Total trainable parameters: {sum(p.numel() for p in self.models.parameters() if p.requires_grad):,}")

        self.epoch = 0
        self.step = 0
        self.current_weighting_strategy = 'size_norm'
        
        logger.info(f"Initialized MultiTaskEngine with {len(task_configs)} tasks on device {self.device}")
        logger.info(f"Task-specific encoders: {'enabled (no bottleneck)' if use_task_encoder else 'disabled'}")
        logger.info(f"Task dataset sizes: {self.task_dataset_sizes}")
        logger.info(f"Task size weights (normalized): {self.task_size_weights.tolist()}")
        for i, (train_set, valid_set, test_set) in enumerate(zip(train_sets, valid_sets, test_sets)):
            logger.info(f"Task {i}: Train={len(train_set)}, Valid={len(valid_set)}, Test={len(test_set)}")
    """
    Converts raw dataset items into batched sequences and target tensors.
    Handles token-level targets (lists per sequence) and sequence-level targets (scalars → tensor).
    Ensures sequence-level tasks always have a 'label' key for compatibility with the model.
    """
    def collate_fn(self, batch):
        sequences = []
        targets_dict = defaultdict(list)
        
        for item in batch:
            if isinstance(item, dict):
                sequences.append(item.get('sequence', ''))
                targets = item.get('targets', {})
                if isinstance(targets, dict):
                    for key, value in targets.items():
                        targets_dict[key].append(value)
            else:
                sequences.append(getattr(item, 'sequence', ''))
        
        processed_targets = {}
        for key, values in targets_dict.items():
            # Check if token-level (list of per-residue labels)
            if values and isinstance(values[0], list):
                # Token-level: keep as list for later padding
                processed_targets[key] = values
            else:
                # Sequence-level: convert to tensor
                try:
                    processed_targets[key] = torch.tensor(values, dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"Could not convert targets for key {key}: {e}")
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
    
    def train(self, num_epoch=4, batch_per_epoch=None, tradeoff=1.0, weighting_strategy='size_norm'):
        from tqdm import tqdm
        
        save_path = "/content/drive/MyDrive/protein_multitask_outputs/multitask_logs"
        os.makedirs(save_path, exist_ok=True)

        logger.info(f"Starting training for {num_epoch} epochs with strategy={weighting_strategy}")
        if weighting_strategy == 'size_norm':
            logger.info(f"  Using dataset-size-normalized weighting: {self.task_size_weights.tolist()}")
        elif weighting_strategy == 'equal':
            logger.info(f"  All tasks equally weighted: 1.0")
        elif weighting_strategy == 'center_task':
            logger.info(f"  Center task weight: 1.0, Auxiliary weight: {tradeoff}")
        elif weighting_strategy == 'boosted':
            logger.info("  Using loss-based dynamic boosting strategy")

        self.models.train()
        self.current_weighting_strategy = weighting_strategy

        train_logs = {
            "per_task_loss": [[] for _ in self.models.task_names],
            "normalized_loss": [[] for _ in self.models.task_names],
            "weighted_loss": [],
            "grad_norm_head": [[] for _ in self.models.task_names],
            "grad_norm_shared": [],
            "task_weights": [],
            "metrics": {}
        }

        for epoch in range(num_epoch):
            logger.info(f"Epoch {epoch + 1}/{num_epoch}")

            dataloaders = []
            for train_set in self.train_sets:
                dataloader = DataLoader(
                    train_set,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_worker,
                    collate_fn=self.collate_fn,
                    pin_memory=False
                )
                dataloaders.append(iter(dataloader))

            if batch_per_epoch is None:
                lengths = [len(DataLoader(ts, batch_size=self.batch_size)) for ts in self.train_sets]
                batch_per_epoch = min(lengths)

            metrics_buffer = []
            progress_bar = tqdm(range(batch_per_epoch), desc=f"Epoch {epoch + 1}")

            for batch_id in progress_bar:
                batches = []

                # --- Get one batch per task ---
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

                # --- Forward pass (returns normalized + raw losses) ---
                per_task_loss_norm, metric, per_task_loss_raw = self.models(batches)

                # --- Update running mean using RAW losses ---
                self.models.update_loss_statistics(per_task_loss_raw)

                # --- Compute dynamic task weights ---
                if weighting_strategy == 'size_norm':
                    weights = self.task_size_weights.to(self.device)

                elif weighting_strategy == 'equal':
                    weights = torch.ones(len(batches), device=self.device) / len(batches)

                elif weighting_strategy == 'center_task':
                    weights = torch.tensor([1.0] + [tradeoff] * (len(batches) - 1), device=self.device)
                    weights = weights / weights.sum()

                elif weighting_strategy == 'boosted':
                    raw = per_task_loss_raw.detach()
                    T = 0.5  # smaller T => more aggressive boosting
                    weights = torch.softmax(raw / T, dim=0)
                    weights = weights.to(self.device)

                else:
                    raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")

                # --- Weighted loss ( normalized losses for gradient stability) ---
                weighted_loss = (per_task_loss_norm * weights).sum()
                weighted_loss = weighted_loss / self.gradient_interval
                weighted_loss.backward()

                if batch_id % 100 == 0:
                    raw_vals = [round(x.item(), 5) for x in per_task_loss_raw]
                    norm_vals = [round(x.item(), 5) for x in per_task_loss_norm]
                    wts = [round(x.item(), 4) for x in weights]
                    tqdm.write(
                        f"[Debug][Epoch {epoch+1} | Batch {batch_id}] "
                        f"Raw={raw_vals}, Norm={norm_vals}, Weights={wts}, WeightedTotal={round(weighted_loss.item(),5)}"
                    )

                for i, loss in enumerate(per_task_loss_raw):
                    train_logs["per_task_loss"][i].append(loss.item())
                    norm_val = per_task_loss_norm[i].item()
                    train_logs["normalized_loss"][i].append(norm_val)
                train_logs["weighted_loss"].append(weighted_loss.item())
                train_logs["task_weights"].append(weights.detach().cpu().tolist())

                # --- Grad norms ---
                for i, model in enumerate(self.models.task_models):
                    norms = [p.grad.norm().item() for p in model.head.parameters() if p.grad is not None]
                    avg_norm = sum(norms)/len(norms) if norms else 0.0
                    train_logs["grad_norm_head"][i].append(avg_norm)

                protbert_params = [
                    p for p in self.models.task_models[0].backbone.protbert.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                if protbert_params:
                    protbert_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in protbert_params])).item()
                    train_logs["grad_norm_shared"].append(protbert_grad_norm)

                # --- Metrics ---
                for metric_name, val in metric.items():
                    if metric_name not in train_logs["metrics"]:
                        train_logs["metrics"][metric_name] = []
                    train_logs["metrics"][metric_name].append(val)

                # --- Gradient clipping ---
                for model in self.models.task_models:
                    torch.nn.utils.clip_grad_norm_(
                        model.task_encoder.parameters() if model.task_encoder else [model.head.parameters()],
                        max_norm=1.0
                    )
                    torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)

                metrics_buffer.append(metric)

                if (batch_id + 1) % self.gradient_interval == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if len(metrics_buffer) > 0:
                        avg_metrics = self._average_metrics(metrics_buffer)
                        progress_bar.set_postfix(avg_metrics)
                    metrics_buffer = []
                    self.step += 1

                if batch_id % 200 == 0:
                    ema = [round(l, 4) for l in self.models.loss_running_mean]
                    tqdm.write(f"[Debug] Running mean losses: {ema}")

            plt.figure(figsize=(8,5))
            for i, name in enumerate(self.models.task_names):
                plt.plot(train_logs["normalized_loss"][i], label=f"{name} norm loss")
            plt.xlabel("Step")
            plt.ylabel("Normalized Loss")
            plt.legend()
            plt.title(f"Epoch {epoch+1} - Per-task normalized losses")
            plt.savefig(os.path.join(save_path, f"epoch{epoch+1}_norm_loss.png"))

            plt.figure(figsize=(8,5))
            for i, name in enumerate(self.models.task_names):
                plt.plot(train_logs["grad_norm_head"][i], label=f"{name} head grad norm")
            plt.plot(train_logs["grad_norm_shared"], label="Shared ProtBERT grad norm", linestyle='--')
            plt.xlabel("Step")
            plt.ylabel("Grad Norm")
            plt.legend()
            plt.title(f"Epoch {epoch+1} - Grad Norms")
            plt.savefig(os.path.join(save_path, f"epoch{epoch+1}_grad_norm.png"))

            with open(os.path.join(save_path, f"train_logs_epoch{epoch+1}.pkl"), "wb") as f:
                pickle.dump(train_logs, f)

            self.epoch += 1
            if self.scheduler:
                self.scheduler.step()
    
    def train_uncertainty_gradnorm_norm(self, num_epoch=4, batch_per_epoch=None, tradeoff=1.0, weighting_strategy='uncertainty_gradnorm', save_path=None,debug=True):
        """
        Training with uncertainty weighting + GradNorm on top-k encoder layers.
        """
        from tqdm import tqdm
        import torch.nn.functional as F
        import os, csv
        import numpy as np
        from collections import defaultdict

        if save_path is None:
            save_path = "./multitask_logs"
        os.makedirs(save_path, exist_ok=True)

        logger.info(f"Starting training for {num_epoch} epochs with strategy={weighting_strategy}")
        self.models.train()
        self.current_weighting_strategy = weighting_strategy

        # Initialize uncertainty params
        if weighting_strategy == 'uncertainty_gradnorm':
            self.gradnorm_alpha = getattr(self, "gradnorm_alpha", 0.12)
            self.initial_losses = None
            self.loss_ema = None  # Exponential moving average of losses
            logger.info(f"Using uncertainty+GradNorm weighting with alpha={self.gradnorm_alpha}")
            logger.info(f"Initial log_sigma values: {[round(x.item(), 3) for x in self.log_sigma]}")

        # Logging containers
        train_logs = {
            "per_task_loss": [[] for _ in self.models.task_names],
            "normalized_loss": [[] for _ in self.models.task_names],
            "weighted_loss": [],
            "grad_norm_head": [[] for _ in self.models.task_names],
            "grad_norm_shared": [],
            "task_weights": [],
            "log_sigma": [[] for _ in self.models.task_names],
            "metrics": {name: defaultdict(list) for name in self.models.task_names}
        }

        # Get top-k encoder layers for GradNorm
        encoder_layers = list(self.models.task_models[0].backbone.protbert.model.encoder.layer)
        topk_layer_patterns = [
            f"encoder.layer.{i}" for i in range(max(0, len(encoder_layers) - 3), len(encoder_layers))
        ]
        logger.info(f"Using last {len(topk_layer_patterns)} encoder layers for GradNorm: {topk_layer_patterns}")

        for epoch in range(num_epoch):
            logger.info(f"\n{'='*70}")
            logger.info(f"EPOCH {epoch + 1}/{num_epoch}")
            logger.info(f"{'='*70}")

            # Create dataloaders for all tasks
            dataloaders = [
                iter(DataLoader(
                    ts,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_worker,
                    collate_fn=self.collate_fn,
                    pin_memory=True
                ))
                for ts in self.train_sets
            ]

            # Determine batches per epoch
            if batch_per_epoch is None:
                lengths = [len(DataLoader(ts, batch_size=self.batch_size)) for ts in self.train_sets]
                batch_per_epoch = min(lengths)
                logger.info(f"Batches per epoch (using min of all tasks): {batch_per_epoch}")

            progress_bar = tqdm(range(batch_per_epoch), desc=f"Epoch {epoch + 1}")
            epoch_loss_recorder = []
            epoch_metrics_per_task = defaultdict(lambda: defaultdict(list))

            for batch_id in progress_bar:
                batches = []

                # --- Get one batch per task ---
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

                    batches.append(self.move_to_device(batch))

                # --- Forward pass ---
                per_task_loss_norm, metrics_flat, per_task_loss_raw = self.models(batches)
                self.models.update_loss_statistics(per_task_loss_raw)

                # --- Record metrics immediately ---
                for task_i, task_name in enumerate(self.models.task_names):
                    task_metrics = {}
                    for metric_name, val in metrics_flat.items():
                        if task_name in metric_name:
                            # Extract metric value
                            metric_key = metric_name.replace(f"{task_name} ", "")
                            task_metrics[metric_key] = val
                            epoch_metrics_per_task[task_name][metric_key].append(val)
                            train_logs["metrics"][task_name][metric_key].append(val)

                # --- Compute weighting and loss ---
                if weighting_strategy == 'uncertainty_gradnorm':
                    # Get precision from uncertainty
                    precision = torch.exp(-self.log_sigma)  # [num_tasks]
                    
                    # Get shared parameters from top-k encoder layers
                    shared_params = []
                    for n, p in self.models.task_models[0].backbone.protbert.model.named_parameters():
                        if p.requires_grad and any(pat in n for pat in topk_layer_patterns):
                            shared_params.append(p)

                    if not shared_params:
                        logger.warning("No shared parameters found for GradNorm!")
                        shared_params = [p for p in self.models.task_models[0].backbone.parameters() if p.requires_grad][:5]

                    # --- Compute gradient norms for each task ---
                    grad_norms = []
                    for t in range(len(per_task_loss_raw)):
                        scalar = precision[t] * per_task_loss_raw[t]
                        grads = torch.autograd.grad(
                            scalar,
                            shared_params,
                            retain_graph=True,
                            allow_unused=True,
                            create_graph=True
                        )
                        usable_grads = [g.detach() for g in grads if g is not None]
                        
                        if usable_grads:
                            g_norm = torch.norm(torch.stack([g.norm() for g in usable_grads]))
                        else:
                            g_norm = torch.tensor(0.0, device=self.device)
                        
                        grad_norms.append(g_norm)

                    grad_norms = torch.stack(grad_norms)  # [num_tasks]

                    # --- Initialize reference losses ---
                    if self.initial_losses is None:
                        self.initial_losses = per_task_loss_raw.detach().clone()
                        self.loss_ema = per_task_loss_raw.detach().clone()
                        logger.info(f"Initialized reference losses: {[round(x.item(), 4) for x in self.initial_losses]}")

                    # --- Update EMA of losses ---
                    ema_decay = 0.95
                    self.loss_ema = ema_decay * self.loss_ema + (1 - ema_decay) * per_task_loss_raw.detach()

                    # --- Compute loss ratios with stability ---
                    loss_ratios = per_task_loss_raw.detach() / (self.loss_ema.clamp(min=1e-8))
                    target = loss_ratios.mean().clamp(min=1e-8, max=1e2)
                    relative_losses = (loss_ratios / target).clamp(min=1e-8, max=10.0)

                    # --- GradNorm loss: balance gradient magnitudes ---
                    grad_norm_mean = grad_norms.mean().clamp(min=1e-8)
                    target_grad = grad_norm_mean * (relative_losses ** self.gradnorm_alpha)
                    gradnorm_loss = F.smooth_l1_loss(grad_norms, target_grad)

                    # --- Uncertainty loss: encourage confident predictions ---
                    uncertainty_term = (precision * per_task_loss_raw + 0.5 * self.log_sigma).mean()

                    # --- Task weights from precision ---
                    weights = (precision / precision.sum()).to(self.device)

                    # --- Total weighted loss ---
                    weighted_loss_norm = (per_task_loss_norm * weights).sum()

                    # --- Regularization coefficients ---
                    reg_uncertainty = 0.01
                    reg_gradnorm = 0.1

                    # --- Total loss ---
                    total_loss = weighted_loss_norm + reg_uncertainty * uncertainty_term + reg_gradnorm * gradnorm_loss

                else:  # Default to size_norm
                    weights = self.task_size_weights.to(self.device)
                    weighted_loss_norm = (per_task_loss_norm * weights).sum()
                    total_loss = weighted_loss_norm
                    gradnorm_loss = torch.tensor(0.0)
                    uncertainty_term = torch.tensor(0.0)

                # --- Backward pass ---
                total_loss_scaled = total_loss / self.gradient_interval
                total_loss_scaled.backward()

                # --- Logging ---
                for i, loss in enumerate(per_task_loss_raw):
                    train_logs["per_task_loss"][i].append(loss.item())
                    train_logs["normalized_loss"][i].append(per_task_loss_norm[i].item())
                    train_logs["log_sigma"][i].append(self.log_sigma[i].item())

                train_logs["weighted_loss"].append(weighted_loss_norm.item())
                train_logs["task_weights"].append(weights.detach().cpu().tolist())

                # --- Gradient norms ---
                for i, model in enumerate(self.models.task_models):
                    head_grads = [p.grad.norm().item() for p in model.head.parameters() if p.grad is not None]
                    avg_head_grad = sum(head_grads) / len(head_grads) if head_grads else 0.0
                    train_logs["grad_norm_head"][i].append(avg_head_grad)

                shared_grad_norms = []
                for p in self.models.task_models[0].backbone.protbert.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        shared_grad_norms.append(p.grad.norm().item())

                if shared_grad_norms:
                    shared_grad_norm = np.mean(shared_grad_norms)
                    train_logs["grad_norm_shared"].append(shared_grad_norm)

                # --- Optimizer step ---
                if (batch_id + 1) % self.gradient_interval == 0:
                    # Gradient clipping
                    for model in self.models.task_models:
                        if model.task_encoder is not None:
                            torch.nn.utils.clip_grad_norm_(model.task_encoder.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.step += 1

                if debug and batch_id % 10 == 0 and batch_id > 0:
                    raw_losses = [round(l.item(), 4) for l in per_task_loss_raw]
                    norms = [round(l.item(), 4) for l in per_task_loss_norm]
                    wts = [round(w.item(), 3) for w in weights]
                    sigmas = [round(self.log_sigma[i].item(), 3) for i in range(len(self.log_sigma))]
                    
                    metrics_str = " | ".join([
                        f"{task_name}: " + ", ".join([
                            f"{k}={round(v, 3)}" 
                            for k, v in dict(list(epoch_metrics_per_task[task_name].items())[-1:]).items()
                        ])
                        for task_name in self.models.task_names
                        if task_name in epoch_metrics_per_task
                    ])

                    logger.info(
                        f"[Epoch {epoch+1} | Batch {batch_id}] "
                        f"Raw={raw_losses} | Norm={norms} | Weights={wts} | σ={sigmas} | "
                        f"Total={round(total_loss.item(), 4)} | {metrics_str}"
                    )

                epoch_loss_recorder.append(total_loss.item())

                avg_metrics_dict = {}
                for task_name in self.models.task_names:
                    for metric_key, values in epoch_metrics_per_task[task_name].items():
                        if values:
                            avg_metrics_dict[f"{task_name}/{metric_key}"] = round(np.mean(values[-5:]), 3)
                
                progress_bar.set_postfix(avg_metrics_dict)

            logger.info(f"\nEpoch {epoch + 1} Summary:")
            
            mean_task_losses = [float(np.mean(v[-batch_per_epoch:])) if v else 0.0 
                                for v in train_logs["per_task_loss"]]
            mean_task_norms = [float(np.mean(v[-batch_per_epoch:])) if v else 0.0 
                            for v in train_logs["normalized_loss"]]
            mean_task_sigmas = [float(np.mean(v[-batch_per_epoch:])) if v else 0.0 
                                for v in train_logs["log_sigma"]]
            
            for i, task_name in enumerate(self.models.task_names):
                logger.info(
                    f"  {task_name}: Loss={mean_task_losses[i]:.4f} | "
                    f"Norm={mean_task_norms[i]:.4f} | σ={mean_task_sigmas[i]:.3f}"
                )

            if train_logs["grad_norm_shared"]:
                mean_grad_shared = float(np.mean(train_logs["grad_norm_shared"][-batch_per_epoch:]))
                logger.info(f"  Shared Encoder Grad Norm: {mean_grad_shared:.4f}")

            logger.info(f"  Task Metrics:")
            for task_name in self.models.task_names:
                for metric_key, values in epoch_metrics_per_task[task_name].items():
                    if values:
                        mean_metric = float(np.mean(values))
                        logger.info(f"    {task_name} {metric_key}: {mean_metric:.4f}")

            # Per-task losses
            plt.figure(figsize=(10, 5))
            for i, name in enumerate(self.models.task_names):
                plt.plot(train_logs["per_task_loss"][i], label=f"{name} (raw)", alpha=0.7)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title(f"Epoch {epoch + 1} - Per-Task Raw Losses")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"epoch{epoch + 1}_task_losses.png"), dpi=100)
            plt.close()

            # Normalized losses
            plt.figure(figsize=(10, 5))
            for i, name in enumerate(self.models.task_names):
                plt.plot(train_logs["normalized_loss"][i], label=f"{name} (normalized)", alpha=0.7)
            plt.xlabel("Step")
            plt.ylabel("Normalized Loss")
            plt.title(f"Epoch {epoch + 1} - Per-Task Normalized Losses")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"epoch{epoch + 1}_normalized_losses.png"), dpi=100)
            plt.close()

            # Log sigma evolution
            plt.figure(figsize=(10, 5))
            for i, name in enumerate(self.models.task_names):
                plt.plot(train_logs["log_sigma"][i], label=f"{name}", alpha=0.7)
            plt.xlabel("Step")
            plt.ylabel("log(σ)")
            plt.title(f"Epoch {epoch + 1} - Uncertainty Evolution")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"epoch{epoch + 1}_uncertainties.png"), dpi=100)
            plt.close()

            # Gradient norms
            plt.figure(figsize=(10, 5))
            for i, name in enumerate(self.models.task_names):
                plt.plot(train_logs["grad_norm_head"][i], label=f"{name} head", alpha=0.7)
            plt.plot(train_logs["grad_norm_shared"], label="Shared encoder", linestyle="--", alpha=0.7)
            plt.xlabel("Step")
            plt.ylabel("Gradient Norm")
            plt.title(f"Epoch {epoch + 1} - Gradient Norms")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"epoch{epoch + 1}_gradnorms.png"), dpi=100)
            plt.close()

            # Task metrics
            for task_name in self.models.task_names:
                if epoch_metrics_per_task[task_name]:
                    plt.figure(figsize=(10, 5))
                    for metric_key, values in epoch_metrics_per_task[task_name].items():
                        plt.plot(values, label=metric_key, alpha=0.7)
                    plt.xlabel("Step")
                    plt.ylabel("Metric Value")
                    plt.title(f"Epoch {epoch + 1} - {task_name} Metrics")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, f"epoch{epoch + 1}_{task_name}_metrics.png"), dpi=100)
                    plt.close()

            csv_path = os.path.join(save_path, "epoch_summary.csv")
            write_header = not os.path.exists(csv_path)
            
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                
                if write_header:
                    header = ["epoch"] + \
                            [f"loss_{n}" for n in self.models.task_names] + \
                            [f"norm_{n}" for n in self.models.task_names] + \
                            [f"sigma_{n}" for n in self.models.task_names]
                    for task_name in self.models.task_names:
                        for metric_key in epoch_metrics_per_task[task_name].keys():
                            header.append(f"{task_name}_{metric_key}")
                    header += ["shared_gradnorm", "weighted_loss_mean"]
                    writer.writerow(header)
                
                row = [epoch + 1] + mean_task_losses + mean_task_norms + mean_task_sigmas
                
                for task_name in self.models.task_names:
                    for metric_key, values in epoch_metrics_per_task[task_name].items():
                        row.append(float(np.mean(values)) if values else 0.0)
                
                row += [
                    float(np.mean(train_logs["grad_norm_shared"])) if train_logs["grad_norm_shared"] else 0.0,
                    float(np.mean(epoch_loss_recorder))
                ]
                writer.writerow(row)

            self.epoch += 1
            if self.scheduler:
                self.scheduler.step()
            
            logger.info(f"Epoch {epoch + 1} completed. Saved to {save_path}\n")

    def _average_metrics(self, metrics_list):
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
                    all_metrics[metric_name] = avg_value
        
        if log:
            logger.info(f"{split.upper()} Results:")
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
        
        for i, model in enumerate(self.models.task_models):
            state[f"task_{i}"] = model.state_dict()
        
        torch.save(state, checkpoint_path)
    
    def load(self, checkpoint_path, load_optimizer=True):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        for i, model in enumerate(self.models.task_models):
            if f"task_{i}" in checkpoint:
                model.load_state_dict(checkpoint[f"task_{i}"])
        
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0) 