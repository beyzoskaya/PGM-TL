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
    preds = preds.squeeze(-1)
    targets = targets.squeeze(-1)
    mse = ((preds - targets) ** 2).mean().item()
    rmse = float(np.sqrt(mse))
    return {'mse': float(mse), 'rmse': rmse}

def classification_metrics(preds, targets, ignore_index=-100):
    if preds.dim() == 3:
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
        pred_labels = preds.argmax(dim=-1)
        targets_flat = targets.view(-1)
        correct = (pred_labels == targets_flat).sum().item()
        total = targets_flat.numel()
        acc = correct / total
        return {'accuracy': float(acc)}

# ----------------------------
# Custom collate_fn
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
def debug_param_update(model, name_filter="encoder"):
    for n, p in model.named_parameters():
        if name_filter in n and p.requires_grad:
            grad = None if p.grad is None else p.grad.abs().mean().item()
            wmean = p.data.abs().mean().item()
            print(f"[GRAD DEBUG] {n}: grad_mean={grad}, weight_mean={wmean}")
            break

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
# MultiTask Engine
# ----------------------------
class MultiTaskEngine:
    def __init__(self, backbone, task_configs, train_sets, valid_sets, test_sets,
                 batch_size=16, device='cuda', alpha=0.99, save_dir=None, max_length=None):
        self.device = device
        self.backbone = backbone.to(device)
        self.task_configs = task_configs
        self.max_length = max_length

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

        hidden_dim = backbone.hidden_size
        self.task_heads = nn.ModuleList([
            nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_dim, cfg['num_labels']))
            for cfg in task_configs
        ]).to(device)

        self.loss_fns = []
        for cfg in task_configs:
            if cfg['type'] == 'regression':
                self.loss_fns.append(nn.MSELoss())
            elif cfg['type'] in ['classification', 'per_residue_classification']:
                self.loss_fns.append(nn.CrossEntropyLoss(ignore_index=-100))
            else:
                raise ValueError(f"Unknown task type {cfg['type']}")

        self.running_losses = torch.ones(len(task_configs), device=device)
        self.alpha = alpha
        self.dynamic_weight_log = []

        self.history = {
            "train_loss": [],
            "val_metrics": [],
            "test_metrics": []
        }

        self.save_dir = ensure_dir(save_dir) if save_dir is not None else None
        self._optimizer = None
        self.best = {"score": [None]*len(task_configs), "path":[None]*len(task_configs)}

    # ---------- forward helpers ----------
    def forward_task(self, embeddings, task_idx):
        tcfg = self.task_configs[task_idx]['type']
        head = self.task_heads[task_idx]
        if tcfg == 'per_residue_classification':
            return head(embeddings)
        else:
            return head(embeddings)

    def compute_loss(self, logits, targets, task_idx):
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
        else:
            return self.loss_fns[task_idx](logits, targets.long())

    def update_dynamic_weights(self):
        inv_losses = 1.0 / (self.running_losses + 1e-8)
        weights = inv_losses / inv_losses.sum()
        self.dynamic_weight_log.append(weights.cpu().numpy().tolist())
        if self.save_dir:
            np.save(os.path.join(self.save_dir, "dynamic_weight_log.npy"), np.array(self.dynamic_weight_log))
        return weights

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

    # ---------- train with debug ----------
    def train_one_epoch(self, optimizer, max_batches_per_loader=None):
        self.backbone.train()
        for head in self.task_heads:
            head.train()

        total_weighted_loss = 0.0
        updates = 0

        # DEBUG: backbone in optimizer
        optim_params = set(p for g in optimizer.param_groups for p in g['params'])
        backbone_params = set(self.backbone.parameters())
        #print("Backbone in optimizer:", backbone_params.issubset(optim_params))
        #print("Backbone requires_grad:", all(p.requires_grad for p in self.backbone.parameters()))

        for task_idx, loader in enumerate(self.train_loaders):
            print(f"\n--- Training Task {task_idx} ---")
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
                embeddings = self.backbone(input_ids, attention_mask, per_residue=per_residue)
                debug_embeddings(task_idx, embeddings)

                logits = self.forward_task(embeddings, task_idx)
                debug_logits(task_idx, logits)

                loss = self.compute_loss(logits, targets, task_idx)

                # Update running loss and dynamic weights
                self.running_losses[task_idx] = self.alpha * self.running_losses[task_idx] + (1.0 - self.alpha) * loss.detach()
                weights = self.update_dynamic_weights()
                weighted_loss = (weights[task_idx].to(self.device) * loss)

                weighted_loss.backward()

                # DEBUG: check gradients
                total_grad_norm = 0.0
                for n, p in self.backbone.named_parameters():
                    if p.grad is not None:
                        gnorm = p.grad.norm().item()
                        total_grad_norm += gnorm
                        #if gnorm == 0.0:
                        #    print(f"WARNING: zero grad for {n}")
                #print("BACKBONE TOTAL GRAD NORM:", total_grad_norm)

                optimizer.step()

                total_weighted_loss += float(weighted_loss.item())
                updates += 1

                if (updates % 50) == 0 or batch_idx == 0:
                    print(f"Task {task_idx} Batch {batch_idx}: raw_loss={loss.item():.4f}, weight={weights[task_idx]:.4f}, weighted_loss={weighted_loss.item():.4f}")
                    #print(f"[DYN DEBUG] running_losses={self.running_losses.cpu().detach().numpy()}, weights={weights.cpu().detach().numpy()}")

        avg_loss = total_weighted_loss / max(1, updates)
        return avg_loss

    def evaluate(self, loaders, split_name="Validation"):
        self.backbone.eval()
        for head in self.task_heads:
            head.eval()

        all_task_metrics = []
        with torch.no_grad():
            for task_idx, loader in enumerate(loaders):
                print(f"Evaluating task {task_idx} on {split_name}")
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

        if self.save_dir:
            fname = f"{split_name.lower()}_metrics_epoch_{len(self.history['train_loss'])+1}.npy"
            np.save(os.path.join(self.save_dir, fname), np.array(all_task_metrics, dtype=object))

        return all_task_metrics

# ----------------------------
# Main script
# ----------------------------
if __name__ == "__main__":
    SAVE_DIR = "/content/drive/MyDrive/protein_multitask_outputs/dynamic_weighting"
    ensure_dir(SAVE_DIR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    thermo_full = Thermostability()
    thermo_train, thermo_valid, thermo_test = thermo_full.split()
    ssp_full = SecondaryStructure()
    ssp_train, ssp_valid, ssp_test = ssp_full.split()
    clf_full = CloningCLF()
    clf_train, clf_valid, clf_test = clf_full.split()

    print("Splits sizes (thermo, ssp, clf):")
    print(len(thermo_train), len(thermo_valid), len(thermo_test))
    print(len(ssp_train), len(ssp_valid), len(ssp_test))
    print(len(clf_train), len(clf_valid), len(clf_test))

    task_configs = [
        {'type': 'regression', 'num_labels': 1},
        {'type': 'per_residue_classification', 'num_labels': 8},
        {'type': 'classification', 'num_labels': 2}
    ]

    backbone = SharedProtBert(lora=True).to(device)
    # Ensure backbone parameters are trainable
    for p in backbone.parameters():
        p.requires_grad = True

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
        max_length=None
    )

    optimizer = optim.Adam(
        list(backbone.parameters()) + list(engine.task_heads.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )

    NUM_EPOCHS = 3
    for epoch in range(1, NUM_EPOCHS+1):
        start = time.time()
        train_loss = engine.train_one_epoch(optimizer)
        epoch_time = time.time() - start
        print(f"\nEpoch {epoch} done in {epoch_time:.1f}s - avg weighted loss: {train_loss:.4f}")
        engine.history["train_loss"].append(train_loss)

        val_metrics = engine.evaluate(engine.valid_loaders, split_name="Validation")
        test_metrics = engine.evaluate(engine.test_loaders, split_name="Test")
        engine.history["val_metrics"].append(val_metrics)
        engine.history["test_metrics"].append(test_metrics)

        for t_idx, m in enumerate(val_metrics):
            if engine.task_configs[t_idx]['type'] == 'regression':
                primary = {'mse': m.get('mse', np.inf)}
            else:
                primary = {'accuracy': m.get('accuracy', 0.0)}
            engine.save_checkpoint(epoch, optimizer=optimizer, name=f"best_task{t_idx}_epoch{epoch}.pt")

        if SAVE_DIR:
            np.save(os.path.join(SAVE_DIR, "train_loss.npy"), np.array(engine.history["train_loss"]))
            np.save(os.path.join(SAVE_DIR, "dynamic_weight_log.npy"), np.array(engine.dynamic_weight_log))
            with open(os.path.join(SAVE_DIR, "history.json"), "w") as fh:
                json.dump(engine.history, fh, indent=2)

    final_test_metrics = engine.evaluate(engine.test_loaders, split_name="Test")
    engine.history["test_metrics"] = final_test_metrics
    np.save(os.path.join(SAVE_DIR, "final_test_metrics.npy"), np.array(final_test_metrics, dtype=object))
    with open(os.path.join(SAVE_DIR, "final_history.json"), "w") as fh:
        json.dump(engine.history, fh, indent=2)

    print("Training finished. Artifacts saved to:", SAVE_DIR)
