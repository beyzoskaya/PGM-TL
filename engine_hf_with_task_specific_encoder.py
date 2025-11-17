import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from protbert_hf import SharedProtBert
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import os
import time
import json

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# ----------------------------
# Utility Metrics
# ----------------------------
def regression_metrics(preds, targets):
    # preds: tensor [B] or [B,1], targets: tensor [B] or [B,1]
    preds = preds.squeeze(-1)
    targets = targets.squeeze(-1)
    mse = ((preds - targets) ** 2).mean().item()
    rmse = float(np.sqrt(mse))
    return {'mse': float(mse), 'rmse': rmse}

def classification_metrics(preds, targets, ignore_index=-100):
    """
    preds: [B, C] or [B, L, C]
    targets: [B] or [B, L] with -100 padding for ignored TOKEN positions
    returns dict {'accuracy': float}
    """
    if preds.dim() == 3:
        # per-token
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
        # sequence-level
        pred_labels = preds.argmax(dim=-1)
        targets_flat = targets.view(-1)
        correct = (pred_labels == targets_flat).sum().item()
        total = targets_flat.numel()
        acc = correct / total
        return {'accuracy': float(acc)}

# ----------------------------
# Custom collate_fn for variable-length sequences
# ----------------------------
def collate_fn(batch, tokenizer=None, max_length=None):
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
        # Pad per-residue targets to match input_ids length
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
        targets = torch.stack(padded_targets, dim=0)  # [B, L]

    return {
        'sequence': input_ids,
        'attention_mask': attention_mask,
        'targets': {'target': targets}
    }

# ----------------------------
# MultiTask Engine with Dynamic Weight Logging
# ----------------------------
class MultiTaskEngine:
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets,
                 batch_size=16, device='cuda', alpha=0.99, save_dir=None, max_length=None):
        self.device = device
        self.backbone = backbone.to(device)
        self.task_configs = task_configs
        self.max_length = max_length

        # Data loaders (pass tokenizer into collate via lambda)
        tokenizer = self.backbone.tokenizer
        self.train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         collate_fn=lambda b, tok=tokenizer: collate_fn(b, tok, max_length=self.max_length))
                              for ds in train_sets]
        self.valid_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         collate_fn=lambda b, tok=tokenizer: collate_fn(b, tok, max_length=self.max_length))
                              for ds in valid_sets]
        self.test_loaders  = [DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         collate_fn=lambda b, tok=tokenizer: collate_fn(b, tok, max_length=self.max_length))
                              for ds in test_sets]

        # Task heads: for per-residue tasks we'll create linear that accepts D->C and is applied token-wise
        hidden_dim = backbone.hidden_size
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, cfg['num_labels'])
            )
            for cfg in task_configs
        ]).to(device)

        # Loss functions
        self.loss_fns = []
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.loss_fns.append(nn.MSELoss())
            elif cfg['type'] in ['classification', 'per_residue_classification']:
                # use ignore_index for per-residue padding
                self.loss_fns.append(nn.CrossEntropyLoss(ignore_index=-100))
            else:
                raise ValueError(f"Unknown task type {cfg['type']}")

        # dynamic weighting bookkeeping
        self.running_losses = torch.ones(len(task_configs), device=device)
        self.alpha = alpha
        self.dynamic_weight_log = []  # list of lists (per update)

        # history and saving
        self.history = {
            "train_loss": [],   # per epoch
            "val_metrics": [],  # per epoch: list of dicts
            "test_metrics": []  # saved after evaluation
        }

        self.save_dir = ensure_dir(save_dir) if save_dir is not None else None

        # placeholders to store optimizer for checkpointing convenience
        self._optimizer = None

        # best model tracking per task (we use simple rule: for regression lower mse is better; for classification higher acc)
        self.best = {
            "score": [None] * len(task_configs),
            "path": [None] * len(task_configs)
        }

    # ---------- forward helpers ----------
    def forward_task(self, embeddings, task_idx):
        """
        embeddings:
          - sequence-level pipeline: [B, D]
          - per-residue pipeline: [B, L, D]
        returns logits shaped:
          - [B, C] for sequence-level
          - [B, L, C] for per-residue
        """
        tcfg = self.task_configs[task_idx]['type']
        head = self.task_heads[task_idx]
        if tcfg == 'per_residue_classification':
            # apply linear to last dim
            logits = head(embeddings)  # shape [B, L, C]
            return logits
        else:
            logits = head(embeddings)  # [B, C]
            return logits

    def compute_loss(self, logits, targets, task_idx):
        tcfg = self.task_configs[task_idx]['type']
        if tcfg == 'regression':
            # logits [B,1] or [B], targets [B]
            # ensure shapes
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            return self.loss_fns[task_idx](logits, targets)
        elif tcfg == 'per_residue_classification':
            # logits [B, L, C], targets [B, L] with -100
            B, L, C = logits.shape
            logits_flat = logits.view(-1, C)
            targets_flat = targets.view(-1)  # long
            return self.loss_fns[task_idx](logits_flat, targets_flat)
        else:
            # sequence-level classification: logits [B, C], targets [B]
            return self.loss_fns[task_idx](logits, targets.long())

    def update_dynamic_weights(self):
        inv_losses = 1.0 / (self.running_losses + 1e-8)
        weights = inv_losses / inv_losses.sum()
        self.dynamic_weight_log.append(weights.cpu().numpy().tolist())

        # save dynamic weight log periodically
        if self.save_dir:
            np.save(os.path.join(self.save_dir, "dynamic_weight_log.npy"), np.array(self.dynamic_weight_log))

        return weights

    # ---------- checkpoint / save ----------
    def save_checkpoint(self, epoch, optimizer=None, name=None):
        if not self.save_dir:
            return

        ckpt = {
            "epoch": epoch,
            "backbone_state": self.backbone.state_dict(),
            "heads_state": self.task_heads.state_dict(),
            "history": self.history,
            "dynamic_weight_log": self.dynamic_weight_log,
            "running_losses": self.running_losses.cpu().tolist()
        }
        if optimizer is not None:
            ckpt["optimizer_state"] = optimizer.state_dict()
        fname = f"checkpoint_epoch_{epoch}.pt" if name is None else name
        path = os.path.join(self.save_dir, fname)
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")
        return path

    def save_best_for_task(self, epoch, task_idx, val_metric, optimizer=None):
        """
        Save a copy when improvement on task_idx observed.
        For regression we expect smaller 'mse'; for classification we expect larger 'accuracy'.
        val_metric: dict e.g. {'mse':..., 'rmse':...} or {'accuracy':...}
        """
        if not self.save_dir:
            return
        score = None
        if self.task_configs[task_idx]['type'] == 'regression':
            score = -float(val_metric.get('mse', np.inf))  # higher is better (negate)
        else:
            score = float(val_metric.get('accuracy', 0.0))

        prev = self.best['score'][task_idx]
        if prev is None or score > prev:
            self.best['score'][task_idx] = score
            name = f"best_task{task_idx}_epoch{epoch}.pt"
            path = self.save_checkpoint(epoch, optimizer=optimizer, name=name)
            self.best['path'][task_idx] = path
            print(f"New best for task {task_idx}: {path} (score={score})")

    # ---------- plotting ----------
    def plot_dynamic_weights(self, save_name="dynamic_weights.png"):
        if len(self.dynamic_weight_log) == 0:
            print("No dynamic weight history to plot")
            return
        arr = np.array(self.dynamic_weight_log)
        plt.figure(figsize=(8,4))
        for i in range(arr.shape[1]):
            plt.plot(arr[:, i], label=f"task_{i}")
        plt.xlabel("Update (batch-level)")
        plt.ylabel("Dynamic weight")
        plt.legend()
        plt.grid(True)
        if self.save_dir:
            out = os.path.join(self.save_dir, save_name)
            plt.savefig(out, bbox_inches="tight")
            print(f"Saved dynamic weight plot: {out}")
        plt.close()

    def plot_metric_history(self, save_prefix="metrics"):
        # history["val_metrics"] is list per-epoch; each element is list of per-task dicts
        if not self.history["val_metrics"]:
            return
        n_epochs = len(self.history["val_metrics"])
        # per-task per-metric plotting
        # gather per-task metric lists
        per_task_metrics = [{} for _ in range(len(self.task_configs))]
        for epoch_metrics in self.history["val_metrics"]:
            # epoch_metrics is list of dicts (one per task)
            for t_idx, mdict in enumerate(epoch_metrics):
                for k, v in mdict.items():
                    per_task_metrics[t_idx].setdefault(k, []).append(v)

        # plot each task-metric
        for t_idx, metrics_dict in enumerate(per_task_metrics):
            for metric_name, values in metrics_dict.items():
                plt.figure(figsize=(6,4))
                plt.plot(range(1, n_epochs+1), values, marker='o')
                plt.title(f"Task {t_idx} - {metric_name}")
                plt.xlabel("Epoch")
                plt.grid(True)
                if self.save_dir:
                    out = os.path.join(self.save_dir, f"{save_prefix}_task{t_idx}_{metric_name}.png")
                    plt.savefig(out, bbox_inches="tight")
                    print(f"Saved metric plot: {out}")
                plt.close()

    # ---------- training / evaluation ----------
    def train(self, num_epochs, optimizer, scheduler=None, save_every=1):
        """
        Train loop for multiple epochs. Only trains and saves checkpoints.
        Validation and test evaluation are handled in main per epoch.
        """
        self._optimizer = optimizer
        for epoch in range(1, num_epochs+1):
            start = time.time()
            train_loss = self.train_one_epoch(optimizer)
            epoch_time = time.time() - start
            print(f"\nEpoch {epoch} done in {epoch_time:.1f}s - avg weighted loss: {train_loss:.4f}")
            self.history["train_loss"].append(train_loss)

            # scheduler step if present
            if scheduler is not None:
                scheduler.step()

            # checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, optimizer=optimizer)

            # persist training history
            if self.save_dir:
                np.save(os.path.join(self.save_dir, "train_loss.npy"), np.array(self.history["train_loss"]))
                np.save(os.path.join(self.save_dir, "dynamic_weight_log.npy"), np.array(self.dynamic_weight_log))
                with open(os.path.join(self.save_dir, "history.json"), "w") as fh:
                    json.dump(self.history, fh, indent=2)

        # after training, produce plots
        self.plot_dynamic_weights()
        self.plot_metric_history()

    def train_one_epoch(self, optimizer, max_batches_per_loader=None):
        self.backbone.train()
        for head in self.task_heads:
            head.train()

        total_weighted_loss = 0.0
        updates = 0

        for task_idx, loader in enumerate(self.train_loaders):
            print(f"\n--- Training Task {task_idx} (loader size: {len(loader)}) ---")
            for batch_idx, batch in enumerate(loader):
                if (max_batches_per_loader is not None) and (batch_idx >= max_batches_per_loader):
                    break

                input_ids = batch['sequence'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets']['target']
                if torch.is_tensor(targets):
                    targets = targets.to(self.device)

                optimizer.zero_grad()

                per_residue = (self.task_configs[task_idx]['type'] == 'per_residue_classification')

                # call backbone: returns either [B, D] or [B, L, D]
                embeddings = self.backbone(input_ids, attention_mask, per_residue=per_residue)

                # for sequence-level tasks ensure embeddings shape [B, D]
                logits = self.forward_task(embeddings, task_idx)

                loss = self.compute_loss(logits, targets, task_idx)

                # update running loss (ema)
                self.running_losses[task_idx] = self.alpha * self.running_losses[task_idx] + (1.0 - self.alpha) * loss.detach()

                # compute dynamic weights (inversely proportional to running_losses)
                weights = self.update_dynamic_weights()
                weighted_loss = (weights[task_idx].to(self.device) * loss)

                weighted_loss.backward()
                optimizer.step()

                total_weighted_loss += float(weighted_loss.item())
                updates += 1

                if (updates % 50) == 0 or batch_idx == 0:
                    print(f"Task {task_idx} Batch {batch_idx}: raw_loss={loss.item():.4f}, weight={weights[task_idx]:.4f}, weighted_loss={weighted_loss.item():.4f}")

        avg_loss = total_weighted_loss / max(1, updates)
        return avg_loss

    def evaluate(self, loaders, split_name="Validation"):
        self.backbone.eval()
        for head in self.task_heads:
            head.eval()

        all_task_metrics = []
        with torch.no_grad():
            for task_idx, loader in enumerate(loaders):
                print(f"Evaluating task {task_idx} on {split_name} (loader size: {len(loader)})")
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
                        metrics = regression_metrics(logits.squeeze(-1).cpu(), targets.cpu())
                    else:
                        metrics = classification_metrics(logits.cpu(), targets.cpu(), ignore_index=-100)

                    accumulated.append(metrics)

                # average metrics across batches (for each metric name)
                if len(accumulated) == 0:
                    avg_metrics = {}
                else:
                    avg_metrics = {}
                    keys = accumulated[0].keys()
                    for k in keys:
                        vals = [a[k] for a in accumulated if not (isinstance(a[k], float) and np.isnan(a[k]))]
                        avg_metrics[k] = float(np.mean(vals)) if len(vals) > 0 else float('nan')
                print(f"{split_name} metrics for Task {task_idx}: {avg_metrics}")
                all_task_metrics.append(avg_metrics)

        # save evaluation metrics
        if self.save_dir:
            fname = f"{split_name.lower()}_metrics_epoch_{len(self.history['train_loss'])+1}.npy"
            np.save(os.path.join(self.save_dir, fname), np.array(all_task_metrics, dtype=object))

        return all_task_metrics

# ----------------------------
# Main usage example
# ----------------------------
if __name__ == "__main__":
    #try:
    #    from google.colab import drive
    #    drive.mount('/content/drive')
    #except Exception:
    #    pass

    SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/dynamic_weighting"
    ensure_dir(SAVE_DIR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------------
    # Load datasets (your dataset classes should provide split() that returns (train,valid,test) Subsets)
    # ----------------------------
    thermo_full = Thermostability()    # uses HF loader
    thermo_train, thermo_valid, thermo_test = thermo_full.split()

    ssp_full = SecondaryStructure()
    ssp_train, ssp_valid, ssp_test = ssp_full.split()

    clf_full = CloningCLF()
    clf_train, clf_valid, clf_test = clf_full.split()

    print("Splits sizes (thermo, ssp, clf):")
    print(len(thermo_train), len(thermo_valid), len(thermo_test))
    print(len(ssp_train), len(ssp_valid), len(ssp_test))
    print(len(clf_train), len(clf_valid), len(clf_test))

    # ----------------------------
    # Task configuration
    # ----------------------------
    task_configs = [
        {'type': 'regression', 'num_labels': 1},
        {'type': 'per_residue_classification', 'num_labels': 8},
        {'type': 'classification', 'num_labels': 2}
    ]

    # ----------------------------
    # Backbone
    # ----------------------------
    backbone = SharedProtBert(lora=False).to(device)

    # ----------------------------
    # Engine
    # ----------------------------
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
        max_length=None  # set if you want to truncate sequences
    )

    # ----------------------------
    # Optimizer + optional scheduler
    # ----------------------------
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(engine.task_heads.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )

    # ----------------------------
    # Run training: adjust epochs as needed
    # ----------------------------
    NUM_EPOCHS = 3

    for epoch in range(1, NUM_EPOCHS+1):
        start = time.time()
        train_loss = engine.train_one_epoch(optimizer)
        epoch_time = time.time() - start
        print(f"\nEpoch {epoch} done in {epoch_time:.1f}s - avg weighted loss: {train_loss:.4f}")
        engine.history["train_loss"].append(train_loss)

        # ----------------------------
        # Evaluate validation and test per epoch
        # ----------------------------
        val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation")
        test_metrics = engine.evaluate(engine.test_loaders, split_name="Test")

        engine.history["val_metrics"].append(val_metrics)
        engine.history["test_metrics"].append(test_metrics)

        # ----------------------------
        # Save best model per task based on validation
        # ----------------------------
        for t_idx, m in enumerate(val_metrics):
            if engine.task_configs[t_idx]['type'] == 'regression':
                primary = {'mse': m.get('mse', np.inf)}
            else:
                primary = {'accuracy': m.get('accuracy', 0.0)}
            engine.save_best_for_task(epoch, t_idx, primary, optimizer=optimizer)

        # ----------------------------
        # Persist history per epoch
        # ----------------------------
        if SAVE_DIR:
            np.save(os.path.join(SAVE_DIR, "train_loss.npy"), np.array(engine.history["train_loss"]))
            np.save(os.path.join(SAVE_DIR, "dynamic_weight_log.npy"), np.array(engine.dynamic_weight_log))
            with open(os.path.join(SAVE_DIR, "history.json"), "w") as fh:
                json.dump(engine.history, fh, indent=2)

    # ----------------------------
    # Final evaluation on test set
    # ----------------------------
    final_test_metrics = engine.evaluate(engine.test_loaders, split_name="Test")
    engine.history["test_metrics"] = final_test_metrics
    np.save(os.path.join(SAVE_DIR, "final_test_metrics.npy"), np.array(final_test_metrics, dtype=object))
    with open(os.path.join(SAVE_DIR, "final_history.json"), "w") as fh:
        json.dump(engine.history, fh, indent=2)

    print("Training finished. Artifacts saved to:", SAVE_DIR)
