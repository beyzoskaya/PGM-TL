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

class TaskModel(nn.Module):
    
    def __init__(self, backbone, task_type, num_labels, task_id, hidden_dim=768, 
                 use_task_encoder=True, encoder_dim=512):
        super().__init__()
        self.backbone = backbone
        self.task_type = task_type
        self.num_labels = num_labels
        self.task_id = task_id
        self.output_dim = backbone.output_dim
        self.use_task_encoder = use_task_encoder

        if use_task_encoder:
            self.task_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Task-specific normalization
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.2)
            )

            self.encoder_norm = nn.LayerNorm(hidden_dim)
        else:
            self.task_encoder = None
            self.encoder_norm = None
        
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )
    
    def forward(self, batch):
        """
        Args:
            batch: {'sequence': [...], 'targets': {...}}
        
        Returns:
            {'logits': [...], 'attention_mask': [...]}
        """
        # Shared backbone forward
        backbone_out = self.backbone(batch)
        
        if self.task_type == 'token_classification':
            # Token-level: [batch, seq_len, hidden]
            features = backbone_out["residue_feature"]
        else:
            # Sequence-level: [batch, hidden]
            features = backbone_out["graph_feature"]
        
        if self.use_task_encoder and self.task_encoder is not None:
            if self.task_type == 'token_classification':
                # For token-level, apply encoder to each token
                batch_size, seq_len, hidden_dim = features.shape
                features = features.reshape(-1, hidden_dim)  # [batch*seq_len, hidden]
                features = self.task_encoder(features)
                features = self.encoder_norm(features)  # Normalize after encoder
                features = features.reshape(batch_size, seq_len, hidden_dim)  # [batch, seq_len, hidden]
            else:
                # For sequence-level, apply encoder to pooled feature
                features = self.task_encoder(features)  # [batch, hidden]
                features = self.encoder_norm(features)  # Normalize after encoder

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
        """Token-level loss: cross-entropy with attention mask"""
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
            model = TaskModel(
                backbone=backbone,
                task_type=config['type'],
                num_labels=config['num_labels'],
                task_id=task_id,
                hidden_dim=backbone.output_dim,
                use_task_encoder=use_task_encoder,
                encoder_dim=encoder_dim
            )
            task_models.append(model)
        
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
        
        self.epoch = 0
        self.step = 0
        self.current_weighting_strategy = 'size_norm'
        
        logger.info(f"Initialized MultiTaskEngine with {len(task_configs)} tasks on device {self.device}")
        logger.info(f"Task-specific encoders: {'enabled (no bottleneck)' if use_task_encoder else 'disabled'}")
        logger.info(f"Task dataset sizes: {self.task_dataset_sizes}")
        logger.info(f"Task size weights (normalized): {self.task_size_weights.tolist()}")
        for i, (train_set, valid_set, test_set) in enumerate(zip(train_sets, valid_sets, test_sets)):
            logger.info(f"Task {i}: Train={len(train_set)}, Valid={len(valid_set)}, Test={len(test_set)}")
    
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
            # Token-level if first element is list-of-values
            if values and isinstance(values[0], list):
                processed_targets[key] = values
            else:
                # Sequence-level scalar targets -> convert to tensor
                try:
                    processed_targets[key] = torch.tensor(values, dtype=torch.float)
                except Exception:
                    processed_targets[key] = values

        # Assign 'label' key for sequence-level tasks
        if hasattr(self, 'sequence_level_tasks'):
            for task_key in self.sequence_level_tasks:
                if task_key in processed_targets and 'label' not in processed_targets:
                    processed_targets['label'] = processed_targets[task_key]

        # --- Debug prints for batch ---
        try:
            print(f"[collate_fn] batch_size={len(sequences)} | target_keys={list(processed_targets.keys())}")
            for k, v in processed_targets.items():
                if isinstance(v, list):
                    print(f"  - {k}: token-level (list of {len(v)} sequences), example len/type -> {type(v[0])}")
                elif isinstance(v, torch.Tensor):
                    print(f"  - {k}: tensor shape={tuple(v.shape)}, dtype={v.dtype}")
                else:
                    print(f"  - {k}: type={type(v)}")
        except Exception:
            pass

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
    
    def train_uncertanity_gradnorm_norm(self, num_epoch=4, batch_per_epoch=None, tradeoff=1.0, weighting_strategy='size_norm'):
        from tqdm import tqdm
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        import os, pickle

        save_path = "/content/drive/MyDrive/protein_multitask_outputs/multitask_logs"
        os.makedirs(save_path, exist_ok=True)

        logger.info(f"Starting training for {num_epoch} epochs with strategy={weighting_strategy}")

        self.models.train()
        self.current_weighting_strategy = weighting_strategy

        # Initialize uncertainty and GradNorm parameters if needed
        if weighting_strategy == 'uncertainty_gradnorm':
            if not hasattr(self, "log_sigma"):
                self.log_sigma = torch.nn.Parameter(torch.zeros(len(self.models.task_names), device=self.device))
                try:
                    self.optimizer.add_param_group({"params": [self.log_sigma], "lr": self.optimizer.param_groups[0]['lr']})
                except Exception as e:
                    logger.warning(f"Could not add log_sigma to optimizer param groups: {e}")
                logger.info(f"  Initialized uncertainty log_sigma parameters for {len(self.models.task_names)} tasks.")
            if not hasattr(self, "gradnorm_alpha"):
                self.gradnorm_alpha = 0.12
            if not hasattr(self, "initial_losses"):
                self.initial_losses = None

        train_logs = {
            "per_task_loss": [[] for _ in self.models.task_names],
            "normalized_loss": [[] for _ in self.models.task_names],
            "weighted_loss": [],
            "grad_norm_head": [[] for _ in self.models.task_names],
            "grad_norm_shared": [],
            "task_weights": [],
            "metrics": {task_name: {} for task_name in self.models.task_names}
        }

        encoder_layers = list(self.models.task_models[0].backbone.protbert.model.encoder.layer)
        num_layers = len(encoder_layers)
        last_k = 3
        topk_layer_patterns = [f"encoder.layer.{i}" for i in range(num_layers - last_k, num_layers)]
        logger.info(f"Using last {last_k} encoder layers for GradNorm: {topk_layer_patterns}")

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
                print(f"[Info] Batch per epoch set to {batch_per_epoch} (based on smallest dataset)")

            metrics_buffer = []
            progress_bar = tqdm(range(batch_per_epoch), desc=f"Epoch {epoch + 1}")

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

                # Forward pass
                per_task_loss_norm, metric_dict, per_task_loss_raw = self.models(batches)

                # Update EMA running mean
                self.models.update_loss_statistics(per_task_loss_raw)

                # Compute weights
                if weighting_strategy == 'size_norm':
                    weights = self.task_size_weights.to(self.device)
                elif weighting_strategy == 'equal':
                    weights = torch.ones(len(batches), device=self.device) / len(batches)
                elif weighting_strategy == 'center_task':
                    weights = torch.tensor([1.0] + [tradeoff] * (len(batches) - 1), device=self.device)
                    weights = weights / weights.sum()
                elif weighting_strategy == 'boosted':
                    T = 0.5
                    weights = torch.softmax(per_task_loss_raw.detach() / T, dim=0).to(self.device)
                elif weighting_strategy == 'uncertainty_gradnorm':
                    precision = torch.exp(-self.log_sigma)
                    weighted_losses_for_gradnorm = precision * per_task_loss_raw + 0.5 * self.log_sigma

                    self.optimizer.zero_grad()
                    weighted_losses_for_gradnorm.sum().backward(retain_graph=True)

                    # Collect top-k shared params grads
                    shared_params = []
                    for n, p in self.models.task_models[0].backbone.protbert.model.named_parameters():
                        if not p.requires_grad:
                            continue
                        if any(pattern in n for pattern in topk_layer_patterns):
                            shared_params.append(p)

                    grad_norms = []
                    for t in range(len(per_task_loss_raw)):
                        grads = [p.grad.detach().clone() for p in shared_params if p.grad is not None]
                        if grads:
                            grad_norms.append(torch.norm(torch.stack([g.norm() for g in grads])))
                        else:
                            grad_norms.append(torch.tensor(0.0, device=self.device))
                    grad_norms = torch.stack(grad_norms) if grad_norms else torch.zeros(len(per_task_loss_raw), device=self.device)

                    if self.initial_losses is None:
                        self.initial_losses = per_task_loss_raw.detach().clone()

                    loss_ratios = per_task_loss_raw.detach() / (self.initial_losses + 1e-12)
                    target = loss_ratios.mean()
                    relative = loss_ratios / (target + 1e-12)
                    gradnorm_loss = torch.mean(torch.abs(grad_norms - grad_norms.mean() * (relative ** self.gradnorm_alpha)))

                    composed_loss_scalar = weighted_losses_for_gradnorm.sum() + 0.1 * gradnorm_loss
                    weights = (precision / precision.sum()).to(self.device)
                    self.optimizer.zero_grad()
                else:
                    raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")

                try:
                    print(f"[Batch {batch_id}] weights = {[round(w.item(),4) for w in weights]}")
                except Exception:
                    pass

                weighted_loss = (per_task_loss_norm * weights).sum() / self.gradient_interval
                weighted_loss.backward()

                # Clip grads & log
                for model in self.models.task_models:
                    torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)

                for i, model in enumerate(self.models.task_models):
                    norms = [p.grad.norm().item() for p in model.head.parameters() if p.grad is not None]
                    avg_norm = sum(norms)/len(norms) if norms else 0.0
                    train_logs["grad_norm_head"][i].append(avg_norm)

                protbert_params = [p for p in self.models.task_models[0].backbone.protbert.model.parameters() if p.requires_grad and p.grad is not None]
                if protbert_params:
                    train_logs["grad_norm_shared"].append(torch.norm(torch.stack([p.grad.norm() for p in protbert_params])).item())

                if (batch_id + 1) % self.gradient_interval == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    metrics_buffer = []
                    self.step += 1

                # Log losses
                for i, loss in enumerate(per_task_loss_raw):
                    train_logs["per_task_loss"][i].append(loss.item())
                    train_logs["normalized_loss"][i].append(per_task_loss_norm[i].item())
                train_logs["weighted_loss"].append(weighted_loss.item())
                train_logs["task_weights"].append(weights.detach().cpu().tolist())

                # Log metrics per task
                for task_id, task_name in enumerate(self.models.task_names):
                    task_metrics = metric_dict.get(task_id, {})
                    for metric_name, val in task_metrics.items():
                        if metric_name not in train_logs["metrics"][task_name]:
                            train_logs["metrics"][task_name][metric_name] = []
                        train_logs["metrics"][task_name][metric_name].append(val)

            # --- Plot per-epoch ---
            plt.figure(figsize=(8,5))
            for i, name in enumerate(self.models.task_names):
                plt.plot(train_logs["normalized_loss"][i], label=f"{name} norm loss")
            plt.xlabel("Step")
            plt.ylabel("Normalized Loss")
            plt.legend()
            plt.title(f"Epoch {epoch+1} - Per-task normalized losses")
            plt.savefig(os.path.join(save_path, f"epoch{epoch+1}_norm_loss.png"))
            plt.close()

            plt.figure(figsize=(8,5))
            for i, name in enumerate(self.models.task_names):
                plt.plot(train_logs["grad_norm_head"][i], label=f"{name} head grad norm")
            if train_logs["grad_norm_shared"]:
                plt.plot(train_logs["grad_norm_shared"], label="Shared ProtBERT grad norm", linestyle='--')
            plt.xlabel("Step")
            plt.ylabel("Grad Norm")
            plt.legend()
            plt.title(f"Epoch {epoch+1} - Grad Norms")
            plt.savefig(os.path.join(save_path, f"epoch{epoch+1}_grad_norm.png"))
            plt.close()

            # --- Save logs ---
            with open(os.path.join(save_path, f"train_logs_epoch{epoch+1}.pkl"), "wb") as f:
                pickle.dump(train_logs, f)

            # --- Print metrics summary per task ---
            print(f"\n=== Epoch {epoch+1} metrics summary ===")
            for task_name in self.models.task_names:
                metrics_dict = train_logs["metrics"][task_name]
                if metrics_dict:
                    metric_str = ", ".join([f"{k}: {sum(v)/len(v):.4f}" for k,v in metrics_dict.items()])
                    print(f"{task_name}: {metric_str}")
            print("="*40)

            self.epoch += 1
            if self.scheduler:
                self.scheduler.step()

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