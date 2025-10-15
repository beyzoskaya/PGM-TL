# sanity_check.py
import os
import sys
import torch
import yaml
import numpy as np
from easydict import EasyDict
from torch.utils.data import Subset
from flip_hf import Thermostability, SecondaryStructure, PeptideHLAMHCAffinity
from engine_hf import create_shared_multitask_model, SharedBackboneModelsWrapper

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to {seed}")

def load_config(config_file="config_hf.yaml"):
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        print(f"[INFO] Loaded config from {config_file}")
        return EasyDict(cfg)
    else:
        print(f"[WARN] Config file not found ({config_file}). Using default minimal config.")
        # Minimal default config
        return EasyDict({
            'model': {'type': 'shared_lora', 'model_name': 'Rostlab/prot_bert_bfd', 'readout': 'pooler', 'freeze_bert': True},
            'datasets': [
                {'type': 'SecondaryStructure', 'path': './data', 'center': True},
                {'type': 'Thermostability', 'path': './data', 'split': 'human_cell', 'center': False},
            ],
            'tasks': [
                {'type': 'token_classification', 'num_labels': 8, 'loss': 'cross_entropy'},
                {'type': 'regression', 'num_labels': 1, 'loss': 'mse'}
            ]
        })

def check_gpu():
    if torch.cuda.is_available():
        print(f"[INFO] CUDA available! GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA not available. Using CPU.")

def test_datasets(cfg, limit_samples=None):
    print("\n[INFO] Loading datasets...")
    train_sets, valid_sets, test_sets = [], [], []
    for i, dataset_cfg in enumerate(cfg.datasets):
        cfg_copy = dataset_cfg.copy()
        dataset_type = cfg_copy.pop('type')
        is_center = cfg_copy.pop('center', False)

        print(f"  Dataset {i}: {dataset_type} (center: {is_center})")
        if dataset_type == 'SecondaryStructure':
            dataset = SecondaryStructure(**cfg_copy)
        elif dataset_type == 'Thermostability':
            dataset = Thermostability(**cfg_copy)
        elif dataset_type == 'PeptideHLAMHCAffinity':
            dataset = PeptideHLAMHCAffinity(**cfg_copy)
        else:
            raise ValueError(f"Unknown dataset: {dataset_type}")

        train_set, valid_set, test_set = dataset.split()

        if limit_samples:
            train_limit = limit_samples.get('train', len(train_set))
            train_set = Subset(train_set, torch.randperm(len(train_set))[:train_limit])
            valid_limit = limit_samples.get('valid', len(valid_set))
            valid_set = Subset(valid_set, torch.randperm(len(valid_set))[:valid_limit])
            test_limit = limit_samples.get('test', len(test_set))
            test_set = Subset(test_set, torch.randperm(len(test_set))[:test_limit])

        print(f"    Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")
        if is_center:
            train_sets = [train_set] + train_sets
            valid_sets = [valid_set] + valid_sets
            test_sets = [test_set] + test_sets
        else:
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)
    return train_sets, valid_sets, test_sets

def test_model(cfg):
    print("\n[INFO] Building shared multitask model...")
    model = create_shared_multitask_model(cfg.tasks, cfg.model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params}, Trainable: {trainable_params}")
    return model

def test_forward_pass(model, dataset, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("\n[INFO] Running dummy forward pass with batch_size=", batch_size)

    sample = dataset[0]
    print("  Sample type:", type(sample))
    print("  Sample content:", sample) 

    if isinstance(sample, dict):
        inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k,v in sample.items()}
        outputs = model(**inputs)
    elif isinstance(sample, tuple) or isinstance(sample, list):
        inputs, labels = sample
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs).unsqueeze(0).to(device)
        else:
            inputs = inputs.unsqueeze(0).to(device)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels).unsqueeze(0).to(device)
        else:
            labels = labels.unsqueeze(0).to(device)
        outputs = model(inputs, labels=labels)
    else:
        raise ValueError(f"Unknown sample type: {type(sample)}")

    print("  Forward pass successful. Output keys:", outputs.keys() if isinstance(outputs, dict) else "output tensor")

def test_checkpoint(model, filename="test_checkpoint.pth"):
    print("\n[INFO] Testing checkpoint save/load...")
    torch.save(model.state_dict(), filename)
    print(f"  Saved checkpoint to {filename}")
    model.load_state_dict(torch.load(filename))
    print(f"  Loaded checkpoint from {filename}")
    os.remove(filename)
    print("  Checkpoint file removed")

if __name__ == "__main__":
    print("[INFO] Starting sanity check script...")
    set_seed(42)
    check_gpu()
    cfg = load_config()

    # Load datasets with very small limit for testing
    train_sets, valid_sets, test_sets = test_datasets(cfg, limit_samples={'train':2,'valid':1,'test':1})

    # Build model
    model = test_model(cfg)

    # Forward pass on a batch from the first train set
    if len(train_sets) > 0:
        test_forward_pass(model, train_sets[0], batch_size=2)

    # Test checkpoint save/load
    test_checkpoint(model)

    print("\n[INFO] Sanity check completed successfully!")
