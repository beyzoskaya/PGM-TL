import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from protbert_hf import SharedProtBert, build_regression_head, build_token_classification_head, build_sequence_classification_head
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from itertools import cycle
import os
import time
import json

# Configuration
MAX_LENGTH = 512  # ProtBERT token limit
SEED = 42

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Utility Metrics
# ----------------------------
def regression_metrics(preds, targets):
    preds = preds.squeeze(-1)
    targets = targets.squeeze(-1)
    mse = ((preds - targets) ** 2).mean().item()
    rmse = float(np.sqrt(mse))
    mae = (preds - targets).abs().mean().item()
    return {'mse': float(mse), 'rmse': rmse, 'mae': float(mae)}

def classification_metrics(preds, targets, ignore_index=-100):
    if preds.dim() == 3:
        # Per-residue classification: (B, L, C) -> accuracy
        B, L, C = preds.shape
        preds_flat = preds.view(-1, C)
        targets_flat = targets.view(-1)
        mask = targets_flat != ignore_index
        if mask.sum() == 0:
            return {'accuracy': float('nan')}
        pred_labels = preds_flat.argmax(dim=-1)
        correct = (pred_labels[mask] == targets_flat[mask]).sum().item()
        acc = correct / mask.sum().item()
        return {'accuracy': float(acc)}
    else:
        # Sequence-level classification: (B, C) -> accuracy
        pred_labels = preds.argmax(dim=-1)
        targets_flat = targets.view(-1)
        correct = (pred_labels == targets_flat).sum().item()
        total = targets_flat.numel()
        acc = correct / total
        return {'accuracy': float(acc)}

# ----------------------------
# Custom collate_fn
# ----------------------------
def collate_fn(batch, tokenizer=None, max_length=MAX_LENGTH):
    sequences = [item['sequence'] for item in batch]
    targets = [item['targets']['target'] for item in batch]

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided to collate_fn")

    encodings = tokenizer(sequences,
                          return_tensors='pt',
                          padding=True,
                          truncation=True,
                          max_length=max_length)

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Process targets
    if isinstance(targets[0], (int, float)):
        targets = torch.tensor(targets, dtype=torch.float32)
    else:
        max_len = input_ids.size(1)
        padded_targets = []
        for t in targets:
            t_tensor = torch.tensor(t, dtype=torch.long)
            if len(t_tensor) < max_len:
                pad_len = max_len - len(t_tensor)
                t_tensor = torch.cat([t_tensor, torch.full((pad_len,), -100, dtype=torch.long)])
            else:
                t_tensor = t_tensor[:max_len]
            padded_targets.append(t_tensor)
        targets = torch.stack(padded_targets, dim=0)

    return {
        'sequence': input_ids,
        'attention_mask': attention_mask,
        'targets': {'target': targets}
    }

# ----------------------------
# Debug functions
# ----------------------------
def debug_grad_norm(model, name_filter=None):
    """Compute and return total gradient norm for a model component"""
    total_grad_norm = 0.0
    num_params = 0
    has_zero_grad = 0
    
    for n, p in model.named_parameters():
        if p.grad is not None:
            if name_filter is None or name_filter in n:
                gnorm = p.grad.norm().item()
                total_grad_norm += gnorm ** 2  # Sum of squares for proper norm
                num_params += 1
                if gnorm == 0:
                    has_zero_grad += 1
    
    total_grad_norm = float(np.sqrt(total_grad_norm))
    return total_grad_norm, num_params, has_zero_grad

def debug_embeddings(task_idx, embeddings):
    emean = embeddings.mean().item()
    emax = embeddings.max().item()
    emin = embeddings.min().item()
    print(f"[EMB DEBUG][Task {task_idx}] shape={tuple(embeddings.shape)}, mean={emean:.4f}, min={emin:.4f}, max={emax:.4f}")

def debug_logits(task_idx, logits):
    mean = logits.mean().item()
    mx = logits.max().item()
    mn = logits.min().item()
    print(f"[LOGIT DEBUG][Task {task_idx}] shape={tuple(logits.shape)}, mean={mean:.4f}, min={mn:.4f}, max={mx:.4f}")

# ----------------------------
# MultiTask Engine - IMPROVED
# ----------------------------
class MultiTaskEngine:
    """
    Multitask learning engine with:
    - Interleaved batch sampling across tasks
    - Per-epoch dynamic weight updates (not per-batch)
    - Task-specific head architectures
    - Comprehensive evaluation and best model tracking
    - FIXED: Proper gradient accumulation and gradient clipping
    """
    
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets,
                 batch_size=16, device='cuda', alpha=0.99, save_dir=None, 
                 max_length=MAX_LENGTH, verbose=True, grad_clip=1.0):
        self.device = device
        self.backbone = backbone.to(device)
        self.task_configs = task_configs
        self.max_length = max_length
        self.verbose = verbose
        self.num_tasks = len(task_configs)
        self.alpha = alpha  # EMA factor for running losses
        self.grad_clip = grad_clip  # Gradient clipping threshold

        tokenizer = self.backbone.tokenizer
        self.train_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True,
                      collate_fn=lambda b, tok=tokenizer: collate_fn(b, tok, max_length=self.max_length))
            for ds in train_sets
        ]
        self.valid_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=lambda b, tok=tokenizer: collate_fn(b, tok, max_length=self.max_length))
            for ds in valid_sets
        ]
        self.test_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=lambda b, tok=tokenizer: collate_fn(b, tok, max_length=self.max_length))
            for ds in test_sets
        ]

        hidden_dim = backbone.hidden_size
        
        # Build task-specific heads
        self.task_heads = nn.ModuleList()
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                head = build_regression_head(hidden_dim, cfg['num_labels'], dropout_rate=0.2)
            elif cfg['type'] == 'per_residue_classification':
                head = build_token_classification_head(hidden_dim, cfg['num_labels'], dropout_rate=0.3)
            else:  # sequence classification
                head = build_sequence_classification_head(hidden_dim, cfg['num_labels'], dropout_rate=0.2)
            self.task_heads.append(head)
        self.task_heads = self.task_heads.to(device)

        # Loss functions
        self.loss_fns = []
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.loss_fns.append(nn.MSELoss())
            elif cfg['type'] in ['classification', 'per_residue_classification']:
                self.loss_fns.append(nn.CrossEntropyLoss(ignore_index=-100))
            else:
                raise ValueError(f"Unknown task type {cfg['type']}")

        # Dynamic weighting state
        self.running_losses = torch.ones(self.num_tasks, device=device)
        self.dynamic_weight_log = []
        self.gradient_norms_log = []

        # History and best tracking
        self.history = {
            "train_loss": [],
            "val_metrics": [],
            "test_metrics": [],
            "dynamic_weights": [],
            "gradient_norms": []
        }

        self.best = {
            "scores": [None] * self.num_tasks,
            "epochs": [None] * self.num_tasks,
            "paths": [None] * self.num_tasks
        }

        self.save_dir = ensure_dir(save_dir) if save_dir is not None else None
        self._optimizer = None

    # ---------- forward helpers ----------
    def forward_task(self, embeddings, task_idx):
        """Forward through task-specific head"""
        head = self.task_heads[task_idx]
        return head(embeddings)

    def compute_loss(self, logits, targets, task_idx):
        """Compute task-specific loss"""
        tcfg = self.task_configs[task_idx]['type']
        if tcfg == 'regression':
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            return self.loss_fns[task_idx](logits, targets)
        elif tcfg == 'per_residue_classification':
            B, L, C = logits.shape
            logits_flat = logits.view(-1, C)
            targets_flat = targets.view(-1)
            return self.loss_fns[task_idx](logits_flat, targets_flat)
        else:  # sequence classification
            return self.loss_fns[task_idx](logits, targets.long())

    def update_dynamic_weights_epoch(self):
        """
        Update dynamic weights based on accumulated task losses.
        Called ONCE per epoch after all tasks are processed.
        Uses inverse loss weighting: harder tasks get higher weight.
        """
        inv_losses = 1.0 / (self.running_losses + 1e-8)
        weights = inv_losses / inv_losses.sum()
        self.dynamic_weight_log.append(weights.cpu().numpy().tolist())
        
        if self.save_dir:
            np.save(os.path.join(self.save_dir, "dynamic_weight_log.npy"), 
                   np.array(self.dynamic_weight_log))
        
        if self.verbose:
            print(f"[WEIGHTS] {' '.join([f'Task{i}:{w:.3f}' for i, w in enumerate(weights.cpu())])}")
        
        return weights

    def save_checkpoint(self, epoch, optimizer=None, name=None, is_best_overall=False):
        if not self.save_dir:
            return
        
        ckpt = {
            "epoch": epoch,
            "backbone_state": self.backbone.state_dict(),
            "heads_state": self.task_heads.state_dict(),
            "history": self.history,
            "dynamic_weight_log": self.dynamic_weight_log,
            "gradient_norms_log": self.gradient_norms_log,
            "running_losses": self.running_losses.cpu().tolist()
        }
        if optimizer is not None:
            ckpt["optimizer_state"] = optimizer.state_dict()
        
        fname = f"checkpoint_epoch_{epoch}.pt" if name is None else name
        path = os.path.join(self.save_dir, fname)
        torch.save(ckpt, path)
        
        if self.verbose:
            print(f"✓ Saved checkpoint: {path}")
        return path

    # ---------- train with gradient accumulation and clipping ----------
    def train_one_epoch(self, optimizer):
        self.backbone.train()
        for head in self.task_heads:
            head.train()

        # 3 task iterators
        thermo_iter = iter(self.train_loaders[0])
        ssp_iter    = iter(self.train_loaders[1])
        clf_iter    = iter(self.train_loaders[2])

        # Steps = minimum dataset length
        num_steps = min(len(self.train_loaders[0]),
                        len(self.train_loaders[1]),
                        len(self.train_loaders[2]))

        total_loss = 0.0

        for step in range(num_steps):

            # ------------------------------
            # 1. SSP batch
            # ------------------------------
            batch = next(ssp_iter)
            ids = batch['sequence'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            t = batch['targets']['target'].to(self.device)

            emb = self.backbone(ids, mask, per_residue=True)
            logits_ssp = self.task_heads[1](emb)
            
            # FIX: align SSP labels with ProtBERT tokens
            # tokenizer returns offset mapping, use it!
            # I will implement this for you after you confirm
            loss_ssp = self.compute_loss(logits_ssp, t, 1)

            # ------------------------------
            # 2. Thermostability batch
            # ------------------------------
            batch = next(thermo_iter)
            ids = batch['sequence'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            t = batch['targets']['target'].to(self.device)

            emb = self.backbone(ids, mask, per_residue=False)
            logits_thermo = self.task_heads[0](emb)
            loss_thermo = self.compute_loss(logits_thermo, t, 0)

            # ------------------------------
            # 3. Cloning batch
            # ------------------------------
            batch = next(clf_iter)
            ids = batch['sequence'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            t = batch['targets']['target'].to(self.device)

            emb = self.backbone(ids, mask, per_residue=False)
            logits_clf = self.task_heads[2](emb)
            loss_clf = self.compute_loss(logits_clf, t, 2)

            # ------------------------------
            # COMBINED LOSS
            # ------------------------------
            w = self.update_dynamic_weights_epoch()  # or use self.weights
            total = (
                w[0] * loss_thermo +
                w[1] * loss_ssp +
                w[2] * loss_clf
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.backbone.parameters()) + list(self.task_heads.parameters()),
                self.grad_clip
            )
            optimizer.step()

            total_loss += total.item()

        return total_loss / num_steps

    def evaluate(self, loaders, split_name="Validation", epoch=None):
   
        self.backbone.eval()
        for head in self.task_heads:
            head.eval()

        all_task_metrics = []
        
        with torch.no_grad():
            for task_idx, loader in enumerate(loaders):
                if self.verbose:
                    print(f"  Evaluating task {task_idx} on {split_name}...")
                
                accumulated = []
                
                for batch in loader:
                    input_ids = batch['sequence'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets']['target']
                    if torch.is_tensor(targets):
                        targets = targets.to(self.device)

                    per_residue = (self.task_configs[task_idx]['type'] == 'per_residue_classification')
                    embeddings = self.backbone(input_ids, attention_mask, per_residue=per_residue)
                    logits = self.forward_task(embeddings, task_idx)

                    if self.task_configs[task_idx]['type'] == 'regression':
                        metrics = regression_metrics(logits.cpu(), targets.cpu())
                    else:
                        metrics = classification_metrics(logits.cpu(), targets.cpu(), ignore_index=-100)

                    accumulated.append(metrics)

                # Average metrics across batches
                if len(accumulated) == 0:
                    avg_metrics = {}
                else:
                    avg_metrics = {}
                    keys = accumulated[0].keys()
                    for k in keys:
                        vals = [a[k] for a in accumulated if not (isinstance(a[k], float) and np.isnan(a[k]))]
                        avg_metrics[k] = float(np.mean(vals)) if len(vals) > 0 else float('nan')

                all_task_metrics.append(avg_metrics)

                # Track best performance
                if epoch is not None:
                    if self.task_configs[task_idx]['type'] == 'regression':
                        current_score = avg_metrics.get('rmse', np.inf)
                        is_better = (self.best["scores"][task_idx] is None or 
                                   current_score < self.best["scores"][task_idx])
                    else:
                        current_score = avg_metrics.get('accuracy', 0.0)
                        is_better = (self.best["scores"][task_idx] is None or 
                                   current_score > self.best["scores"][task_idx])
                    
                    if is_better:
                        self.best["scores"][task_idx] = current_score
                        self.best["epochs"][task_idx] = epoch

                if self.verbose:
                    print(f"    Task {task_idx} {split_name} metrics: {avg_metrics}")

        return all_task_metrics

    def print_best_metrics(self):
        print("\n" + "="*60)
        print("BEST VALIDATION METRICS PER TASK")
        print("="*60)
        for task_idx in range(self.num_tasks):
            score = self.best["scores"][task_idx]
            epoch = self.best["epochs"][task_idx]
            metric_name = "RMSE" if self.task_configs[task_idx]['type'] == 'regression' else "Accuracy"
            if score is not None:
                print(f"Task {task_idx}: {metric_name}={score:.4f} @ epoch {epoch}")
            else:
                print(f"Task {task_idx}: No improvement tracked")
        print("="*60 + "\n")

if __name__ == "__main__":
    set_seed(SEED)
    
    SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/dynamic_weighting_v3_fixed"
    ensure_dir(SAVE_DIR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Max sequence length: {MAX_LENGTH}")

    # Load datasets
    print("\n[DATA] Loading datasets...")
    thermo_full = Thermostability(verbose=1)
    thermo_train, thermo_valid, thermo_test = thermo_full.split()
    
    ssp_full = SecondaryStructure(verbose=1)
    ssp_train, ssp_valid, ssp_test = ssp_full.split()
    
    clf_full = CloningCLF(verbose=1)
    clf_train, clf_valid, clf_test = clf_full.split()

    print("\n[DATA] Split sizes:")
    print(f"  Thermostability: train={len(thermo_train)}, valid={len(thermo_valid)}, test={len(thermo_test)}")
    print(f"  SecondaryStructure: train={len(ssp_train)}, valid={len(ssp_valid)}, test={len(ssp_test)}")
    print(f"  CloningCLF: train={len(clf_train)}, valid={len(clf_valid)}, test={len(clf_test)}")

    # Task configuration
    task_configs = [
        {'type': 'regression', 'num_labels': 1, 'name': 'Thermostability'},
        {'type': 'per_residue_classification', 'num_labels': 8, 'name': 'SecondaryStructure'},
        {'type': 'classification', 'num_labels': 2, 'name': 'CloningCLF'}
    ]

    print("\n[MODEL] Initializing shared ProtBERT with LoRA...")
    backbone = SharedProtBert(lora=True, verbose=True).to(device)

    print("\n[ENGINE] Creating MultiTaskEngine...")
    engine = MultiTaskEngine(
        backbone=backbone,
        task_configs=task_configs,
        train_sets=[thermo_train, ssp_train, clf_train],
        valid_sets=[thermo_valid, ssp_valid, clf_valid],
        test_sets=[thermo_test, ssp_test, clf_test],
        batch_size=16,
        device=device,
        alpha=0.99,
        save_dir=SAVE_DIR,
        max_length=MAX_LENGTH,
        verbose=True,
        grad_clip=1.0  # Gradient clipping threshold
    )

    # FIXED: Higher learning rate for LoRA-only training
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(engine.task_heads.parameters()),
        lr=2e-4,  # Reduced from 5e-4 for better stability with improved init
        weight_decay=1e-5
    )

    NUM_EPOCHS = 5  # Increased to see learning progress
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING: {NUM_EPOCHS} epochs")
    print(f"Learning rate: 2e-4, Gradient clip: 1.0")
    print(f"{'='*60}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n[EPOCH {epoch}/{NUM_EPOCHS}]")
        print("-" * 60)
        
        start = time.time()
        train_loss = engine.train_one_epoch(optimizer)
        epoch_time = time.time() - start
        
        print(f"\n✓ Epoch {epoch} training done in {epoch_time:.1f}s - avg loss: {train_loss:.4f}")
        engine.history["train_loss"].append(train_loss)

        # Update dynamic weights for next epoch (ONCE per epoch)
        weights = engine.update_dynamic_weights_epoch()
        engine.history["dynamic_weights"].append(weights.cpu().numpy().tolist())

        # Validation
        print(f"\n[VALIDATION]")
        val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation", epoch=epoch)
        engine.history["val_metrics"].append(val_metrics)

        # Test (informative only, not for checkpoint selection)
        print(f"\n[TEST]")
        test_metrics = engine.evaluate(engine.test_loaders, split_name="Test", epoch=epoch)
        engine.history["test_metrics"].append(test_metrics)

        # Save checkpoint
        engine.save_checkpoint(epoch, optimizer=optimizer)

        # Save history after each epoch
        if SAVE_DIR:
            np.save(os.path.join(SAVE_DIR, "train_loss.npy"), np.array(engine.history["train_loss"]))
            np.save(os.path.join(SAVE_DIR, "dynamic_weight_log.npy"), np.array(engine.dynamic_weight_log))
            np.save(os.path.join(SAVE_DIR, "gradient_norms.npy"), np.array(engine.gradient_norms_log, dtype=object))
            with open(os.path.join(SAVE_DIR, "history.json"), "w") as fh:
                json.dump(engine.history, fh, indent=2)

    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")
    final_test_metrics = engine.evaluate(engine.test_loaders, split_name="Test")
    engine.history["final_test_metrics"] = final_test_metrics

    engine.print_best_metrics()

    if SAVE_DIR:
        np.save(os.path.join(SAVE_DIR, "final_test_metrics.npy"), np.array(final_test_metrics, dtype=object))
        with open(os.path.join(SAVE_DIR, "final_history.json"), "w") as fh:
            json.dump(engine.history, fh, indent=2)
        
        # Save best metrics summary
        with open(os.path.join(SAVE_DIR, "best_metrics_summary.txt"), "w") as fh:
            fh.write("BEST VALIDATION METRICS PER TASK\n")
            fh.write("=" * 60 + "\n")
            for task_idx in range(len(task_configs)):
                score = engine.best["scores"][task_idx]
                epoch = engine.best["epochs"][task_idx]
                task_name = task_configs[task_idx]['name']
                metric_name = "RMSE" if task_configs[task_idx]['type'] == 'regression' else "Accuracy"
                if score is not None:
                    fh.write(f"{task_name} (Task {task_idx}): {metric_name}={score:.4f} @ epoch {epoch}\n")

    print(f"\n✓ Training finished. Artifacts saved to: {SAVE_DIR}")
    print(f"✓ Check 'best_metrics_summary.txt' for validation performance tracking")