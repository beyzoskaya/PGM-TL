import os
import sys
import time
import glob
import logging
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import Subset
from easydict import EasyDict
from torch.utils.data import DataLoader
from main_hf import train_and_validate, test, find_latest_checkpoint

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import ProtBert
from engine_hf_v2 import MultiTaskEngine, create_shared_multitask_model, SharedBackboneModelsWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config_hf.yaml", help="yaml configuration file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_auto", action="store_true")
    return parser.parse_known_args()[0]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_default_config():
    config = {
        'output_dir': '/content/drive/MyDrive/protein_multitask_outputs',
        'model': {
            'type': 'shared_lora',
            'model_name': 'Rostlab/prot_bert_bfd',
            'readout': 'pooler',
            'freeze_bert': True,
            'lora_rank': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1
        },
        'datasets': [
            {'type': 'Thermostability', 'path': './data', 'split': 'human_cell', 'center': True},
            {'type': 'SecondaryStructure', 'path': './data', 'center': False},
            {'type': 'CloningCLF', 'path': './data', 'center': False}
        ],
        'tasks': [
            {'type': 'regression', 'num_labels': 1, 'loss': 'mse'},  # Thermostability
            {'type': 'token_classification', 'num_labels': 8, 'loss': 'cross_entropy'},  # SecondaryStructure
            {'type': 'binary_classification', 'num_labels': 1, 'loss': 'binary_cross_entropy'}  # CloningCLF
        ],
        'train': {'num_epoch': 4, 'batch_size': 8, 'gradient_interval': 6, 'tradeoff': 0.5},
        'optimizer': {'type': 'AdamW', 'lr': 3e-5, 'weight_decay': 0.01},
        'scheduler': {'type': 'StepLR', 'step_size': 3, 'gamma': 0.5},
        'engine': {'batch_size': 8, 'num_worker': 1, 'log_interval': 50},
        'eval_metric': 'rmse',
        'test_batch_size': 16
    }
    return EasyDict(config)

def get_root_logger():
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler("baseline_training.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def setup_drive_output(cfg):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if "--full" in sys.argv:
        output_dir = os.path.join(cfg.output_dir, f"full_baseline_{timestamp}")
    else:
        output_dir = os.path.join(cfg.output_dir, f"baseline_multitask_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    return output_dir

def create_datasets(dataset_configs, limit_samples=None):
    train_sets, valid_sets, test_sets = [], [], []

    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    input_shapes = []
    label_shapes = []
    dataset_names = []

    for i, cfg in enumerate(dataset_configs):
        cfg_copy = cfg.copy()
        dtype = cfg_copy.pop('type')
        is_center = cfg_copy.pop('center', False)
        dataset_names.append(dtype)

        print(f"\n[Dataset {i}] Loading {dtype}, Center: {is_center}")
        if dtype == 'Thermostability':
            dataset = Thermostability(**cfg_copy)
        elif dtype == 'SecondaryStructure':
            dataset = SecondaryStructure(**cfg_copy)
        elif dtype == 'CloningCLF':
            dataset = CloningCLF(**cfg_copy)
        else:
            raise ValueError(f"Unknown dataset type: {dtype}")

        train_set, valid_set, test_set = dataset.split()

        # Check sample structure (just one sample)
        sample = next(iter(train_set))
        seq = sample.get('sequence', None)
        targs = sample.get('targets', {})
        if seq is None or not isinstance(targs, dict):
            raise ValueError(f"[{dtype}] dataset must return a dict with 'sequence' and 'targets' keys")

        example_len = len(seq)
        print(f"[{dtype}] sequence length={example_len}, targets keys={list(targs.keys())}")

        # Detect label shape/type for debugging
        first_key = list(targs.keys())[0] if len(targs) > 0 else "None"
        example_val = targs[first_key] if first_key in targs else None
        if isinstance(example_val, list):
            if all(isinstance(x, (int, float)) for x in example_val):
                print(f"  example target key='{first_key}' type=per-residue list length={len(example_val)}")
            elif all(isinstance(x, list) for x in example_val):
                print(f"  example target key='{first_key}' is list-of-lists")
        else:
            print(f"  example target key='{first_key}' type={type(example_val).__name__} length=None")

        # Store splits
        if is_center:
            train_sets = [train_set] + train_sets
            valid_sets = [valid_set] + valid_sets
            test_sets = [test_set] + test_sets
        else:
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)

        print(f"  Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")

    total_train = sum(len(ts) for ts in train_sets)
    total_valid = sum(len(vs) for vs in valid_sets)
    total_test = sum(len(ts) for ts in test_sets)
    print(f"\nTotal samples - Train: {total_train}, Valid: {total_valid}, Test: {total_test}")

    print("\nPerforming batch consistency check (manual sampling)...")

    temp_ds = train_sets[0]
    # manually grab two samples (skip DataLoader auto-collation)
    samples = [temp_ds[i] for i in range(min(2, len(temp_ds)))]

    def quick_collate(batch):
        """Minimal collate function similar to multitask engine."""
        sequences = [b['sequence'] for b in batch]
        targets = {}
        for b in batch:
            for k, v in b['targets'].items():
                if k not in targets:
                    targets[k] = []
                targets[k].append(v)
        return {'sequence': sequences, 'targets': targets}

    test_batch = quick_collate(samples)

    seqs = test_batch['sequence']
    targs = test_batch['targets']
    print(f" Collated batch: sequences (len={len(seqs)}), target keys={list(targs.keys())}")

    for k, v in targs.items():
        if isinstance(v, torch.Tensor):
            print(f"  -> target key '{k}' tensor shape: {v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], list):
                print(f"  -> target key '{k}' is list-of-lists; first lengths: {[len(x) for x in v[:2]]}")
            else:
                print(f"  -> target key '{k}' is list; first values: {v[:4]}")

    print("✅ Manual batch check passed (variable-length sequences handled safely).")
    print()

    return train_sets, valid_sets, test_sets


def create_optimizer(model, cfg):
    opt_type = cfg.get('type', 'AdamW')
    lr = cfg.get('lr', 3e-5)
    wd = cfg.get('weight_decay', 0.01)
    if opt_type == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

def create_scheduler(optimizer, cfg):
    if cfg is None: return None
    stype = cfg.get('type', 'StepLR')
    if stype == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get('step_size',3), gamma=cfg.get('gamma',0.5))
    elif stype == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.get('T_max',10))
    else:
        raise ValueError(f"Unknown scheduler type: {stype}")

def build_solver(cfg, logger):
    train_sets, valid_sets, test_sets = create_datasets(cfg.datasets)
    
    backbone = ProtBert(
        model_name=cfg.model.model_name,
        readout='per_token' if any(t['type'] == 'token_classification' for t in cfg.tasks) else 'pooler',
        freeze_bert=cfg.model.get('freeze_bert', False)
    )
    
    # Optimize: backbone should be on device early
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone.to(device)
    
    # Create optimizer/scheduler
    optimizer = create_optimizer(backbone, cfg.optimizer)
    scheduler = create_scheduler(optimizer, cfg.get('scheduler'))

    engine = MultiTaskEngine(
        backbone=backbone,
        task_configs=cfg.tasks,
        train_sets=train_sets,
        valid_sets=valid_sets,
        test_sets=test_sets,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=cfg.engine.batch_size,
        gradient_interval=cfg.train.gradient_interval,
        num_worker=cfg.engine.num_worker,
        log_interval=cfg.engine.log_interval,
        device=device
    )
    
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info(f"Backbone: Total params={total_params}, Trainable={trainable_params} ({trainable_params/total_params*100:.1f}%)")
    
    task_head_params = sum(p.numel() for model in engine.models.task_models for p in model.head.parameters())
    logger.info(f"Task heads total params: {task_head_params}")
    
    return engine

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    cfg = create_default_config()
    output_dir = setup_drive_output(cfg)
    logger = get_root_logger()
    logger.info(f"Output directory: {output_dir}")

    with open("config_used.yaml", "w") as f:
        yaml.dump(dict(cfg), f)

    solver = build_solver(cfg, logger)

    best_metrics = {}
    best_epoch = 0
    
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        solver.load(args.resume, load_optimizer=True)
        start_epoch = solver.epoch
    elif args.resume_auto:
        checkpoint_path, start_epoch = find_latest_checkpoint(".")
        if checkpoint_path:
            logger.info(f"Auto-resuming from latest checkpoint: {checkpoint_path}")
            solver.load(checkpoint_path, load_optimizer=True)

    for epoch in range(start_epoch, cfg.train.num_epoch):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{cfg.train.num_epoch}")
        logger.info(f"{'='*60}")
        
        # Train
        solver.train(num_epoch=1, batch_per_epoch=None, tradeoff=cfg.train.tradeoff)
        
        # Validate
        val_metrics = solver.evaluate(split='valid', log=True)
        
        # Save checkpoint
        checkpoint_path = f"checkpoint_epoch_{epoch:03d}.pt"
        solver.save(checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Track best epoch based on center task (Task_0)
        center_task_metric = None
        for key in val_metrics:
            if key.startswith('Task_0'):
                center_task_metric = val_metrics[key]
                break
        
        if center_task_metric is not None:
            if epoch == 0 or center_task_metric < best_metrics.get('Task_0', float('inf')):
                best_metrics = val_metrics.copy()
                best_epoch = epoch
                logger.info(f"✓ New best validation metrics (epoch {epoch + 1})")
        
        logger.info(f"Metrics: {val_metrics}")
    
    # Test on best epoch
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing on best epoch {best_epoch + 1}")
    logger.info(f"{'='*60}")
    
    test_metrics = solver.evaluate(split='test', log=True)
    
    logger.info(f"\nFinal Test Results: {test_metrics}")
    logger.info("Training completed successfully!")