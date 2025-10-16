import os
import sys
import math
import pprint
import shutil
import logging
import argparse
import numpy as np
import yaml
from easydict import EasyDict
import time
import glob

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Subset

from flip_hf import create_dataset, Thermostability, SecondaryStructure, CloningCLF
from engine_hf import MultiTaskEngine
from engine_hf import create_shared_multitask_model, SharedBackboneModelsWrapper
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config_hf.yaml", help="yaml configuration file")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--use_contrastive", action="store_true", help="use contrastive loss")
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--phase1", action="store_true", help="quick validation with limited data")
    parser.add_argument("--phase2", action="store_true", help="full baseline training")
    parser.add_argument("--full", action="store_true", help="full baseline training")
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_valid", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--resume_auto", action="store_true", help="auto-resume latest checkpoint")
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
        'output_dir': './outputs',
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
            {'type': 'SecondaryStructure', 'path': './data', 'center': True},
            {'type': 'Thermostability', 'path': './data', 'split': 'human_cell', 'center': False},
            {'type': 'CloningCLF', 'path': './data', 'center': False}  # updated
        ],
        'tasks': [
            {'type': 'token_classification', 'num_labels': 8, 'loss': 'cross_entropy'},
            {'type': 'regression', 'num_labels': 1, 'loss': 'mse'},
            {'type': 'binary_classification', 'num_labels': 1, 'loss': 'binary_cross_entropy'}  # updated
        ],

        'train': {'num_epoch': 6, 'batch_size': 8, 'gradient_interval': 4, 'tradeoff': 0.5},
        'optimizer': {'type': 'AdamW', 'lr': 2e-5, 'weight_decay': 0.01},
        'scheduler': {'type': 'StepLR', 'step_size': 3, 'gamma': 0.5},
        'engine': {'batch_size': 8, 'num_worker': 1, 'log_interval': 50},
        'eval_metric': 'accuracy',
        'test_batch_size': 16
    }
    return EasyDict(config)

def setup_drive_output(cfg):
    if not os.path.exists('/content/drive/MyDrive'):
        raise RuntimeError("Google Drive not mounted.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_drive_path = "/content/drive/MyDrive/protein_multitask_outputs"

    if "--full" in sys.argv:
        output_dir = f"{base_drive_path}/full_baseline_{timestamp}"
    else:
        output_dir = f"{base_drive_path}/baseline_multitask_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    print(f"Output directory: {output_dir}")
    return output_dir

def load_config(config_file):
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return EasyDict(yaml.safe_load(f))
    else:
        print(f"Config {config_file} not found. Using default.")
        return create_default_config()

def create_datasets(dataset_configs, limit_samples=None):
    train_sets, valid_sets, test_sets = [], [], []

    print("="*60)
    print("LOADING DATASETS")
    print("="*60)

    for i, cfg in enumerate(dataset_configs):
        cfg_copy = cfg.copy()
        dtype = cfg_copy.pop('type')
        is_center = cfg_copy.pop('center', False)

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
        original_sizes = (len(train_set), len(valid_set), len(test_set))

        if limit_samples and not is_center:
            train_limit = limit_samples.get('train', len(train_set))
            valid_limit = limit_samples.get('valid', len(valid_set))
            test_limit = limit_samples.get('test', len(test_set))

            if len(train_set) > train_limit:
                train_set = Subset(train_set, torch.randperm(len(train_set))[:train_limit])
            if len(valid_set) > valid_limit:
                valid_set = Subset(valid_set, torch.randperm(len(valid_set))[:valid_limit])
            if len(test_set) > test_limit:
                test_set = Subset(test_set, torch.randperm(len(test_set))[:test_limit])

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

    return train_sets, valid_sets, test_sets

def create_optimizer(model, cfg):
    opt_type = cfg.get('type', 'AdamW')
    lr = cfg.get('lr', 2e-5)
    wd = cfg.get('weight_decay', 0.01)

    if opt_type == 'AdamW':
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'Adam':
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

def create_scheduler(optimizer, cfg):
    if cfg is None:
        return None
    stype = cfg.get('type', 'StepLR')
    if stype == 'StepLR':
        return StepLR(optimizer, step_size=cfg.get('step_size', 3), gamma=cfg.get('gamma', 0.5))
    elif stype == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=cfg.get('T_max', 10))
    else:
        raise ValueError(f"Unknown scheduler type: {stype}")

def build_solver(cfg, logger, limit_samples=None):
    # Create datasets
    train_sets, valid_sets, test_sets = create_datasets(cfg.datasets, limit_samples)

    logger.info("=" * 60)
    logger.info("CREATING SHARED BACKBONE MODEL")
    logger.info("=" * 60)

    shared_model = create_shared_multitask_model(cfg.tasks, cfg.model)

    total_params = sum(p.numel() for p in shared_model.parameters())
    trainable_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f"Model Analysis: Total params={total_params:,}, Trainable={trainable_params:,} "
                f"({trainable_params/total_params*100:.1f}%), Frozen={frozen_params:,}")

    for task_name, head in shared_model.task_heads.items():
        head_params = sum(p.numel() for p in head.parameters())
        logger.info(f"Task head '{task_name}': {head_params:,} parameters")

    optimizer = create_optimizer(shared_model, cfg.optimizer)
    scheduler = create_scheduler(optimizer, cfg.get('scheduler'))

    logger.info(f"Optimizer: {optimizer.__class__.__name__} with lr={cfg.optimizer.get('lr', 2e-5)}")
    if scheduler:
        logger.info(f"Scheduler: {scheduler.__class__.__name__}")

    task_names = [f"Task_{i}" for i in range(len(cfg.tasks))]
    models_wrapper = SharedBackboneModelsWrapper(shared_model, task_names)

    class SharedBackboneMultiTaskEngine(MultiTaskEngine):
        def __init__(self, shared_model, train_sets, valid_sets, test_sets, optimizer, scheduler=None, **kwargs):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.batch_size = kwargs.get('batch_size', 4)
            self.gradient_interval = kwargs.get('gradient_interval', 1)
            self.num_worker = kwargs.get('num_worker', 0)
            self.log_interval = kwargs.get('log_interval', 100)

            self.models = models_wrapper
            self.models.to(self.device)

            self.train_sets = train_sets
            self.valid_sets = valid_sets
            self.test_sets = test_sets
            self.optimizer = optimizer
            self.scheduler = scheduler

            self.epoch = 0
            self.step = 0

            logger.info(f"Initialized SharedBackbone MultiTaskEngine with {len(cfg.tasks)} tasks")

    solver = SharedBackboneMultiTaskEngine(
        shared_model=shared_model,
        train_sets=train_sets,
        valid_sets=valid_sets,
        test_sets=test_sets,
        optimizer=optimizer,
        scheduler=scheduler,
        **cfg.engine
    )

    logger.info(f"Device: {solver.device}, Batch size: {solver.batch_size}, "
                f"Gradient interval: {solver.gradient_interval}, "
                f"Effective batch size: {solver.batch_size * solver.gradient_interval}")

    return solver

def find_latest_checkpoint(output_dir="."):
    checkpoint_pattern = os.path.join(output_dir, "baseline_model_epoch_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        return None, 0

    epochs = []
    for cp in checkpoints:
        try:
            epoch_num = int(cp.split('_epoch_')[1].split('.pth')[0])
            epochs.append((epoch_num, cp))
        except Exception:
            continue

    if not epochs:
        return None, 0

    latest_epoch, latest_checkpoint = max(epochs, key=lambda x: x[0])
    return latest_checkpoint, latest_epoch

def train_and_validate(cfg, solver, logger):
    args = parse_args()
    best_score = float("-inf")
    best_epoch = -1
    start_epoch = 0

    # Handle resume
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        solver.load(args.resume, load_optimizer=True)
        start_epoch = solver.epoch
        logger.info(f"Resumed from epoch {start_epoch}")
    elif args.resume_auto:
        checkpoint_path, start_epoch = find_latest_checkpoint(".")
        if checkpoint_path:
            logger.info(f"Auto-resuming from checkpoint: {checkpoint_path}")
            solver.load(checkpoint_path, load_optimizer=True)
        else:
            logger.info("No checkpoint found for auto-resume, starting fresh")

    if cfg.train.num_epoch <= start_epoch:
        logger.info("Training already completed or no epochs remaining")
        return solver, best_epoch

    remaining_epochs = cfg.train.num_epoch - start_epoch
    total_train_samples = sum(len(ts) for ts in solver.train_sets)
    batches_per_epoch = total_train_samples // (solver.batch_size * solver.gradient_interval)
    est_time_per_epoch = batches_per_epoch * 0.5 / 60  # rough estimate

    logger.info("="*80)
    logger.info("TRAINING PHASE STARTED")
    logger.info(f"Target epochs: {cfg.train.num_epoch}, Remaining: {remaining_epochs}")
    logger.info(f"Effective batch size: {solver.batch_size * solver.gradient_interval}, "
                f"Batches/epoch: ~{batches_per_epoch}, Estimated time/epoch: ~{est_time_per_epoch:.1f} mins")
    logger.info("="*80)

    for epoch_idx in range(start_epoch, cfg.train.num_epoch):
        logger.info(f"\nTraining epoch {epoch_idx+1}/{cfg.train.num_epoch}")
        solver.train(num_epoch=1, tradeoff=cfg.train.get('tradeoff', 1.0))

        # Save checkpoint
        checkpoint_path = f"baseline_model_epoch_{solver.epoch}.pth"
        solver.save(checkpoint_path)
        solver.save("latest_checkpoint.pth")
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Validation
        metrics = solver.evaluate("valid")
        logger.info("-" * 60)
        logger.info(f"Validation Metrics - Epoch {solver.epoch}")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("-" * 60)

        # Track best center task
        score = []
        for k, v in metrics.items():
            if ("Task_0" in k or "Center" in k) and cfg.eval_metric in k.lower():
                if "mse" in cfg.eval_metric or "rmse" in cfg.eval_metric:
                    score.append(-v)
                else:
                    score.append(v)

        if score:
            current_score = sum(score) / len(score)
            if current_score > best_score:
                best_score = current_score
                best_epoch = solver.epoch
                logger.info(f"NEW BEST CENTER TASK SCORE: {best_score:.4f} at epoch {best_epoch}")
                solver.save(f"best_baseline_model_epoch_{best_epoch}.pth")

        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory: {memory_used:.1f}GB used, {memory_reserved:.1f}GB reserved")

        if solver.scheduler:
            current_lr = solver.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.2e}")

    # Load best model
    if best_epoch > 0:
        best_model_path = f"best_baseline_model_epoch_{best_epoch}.pth"
        if os.path.exists(best_model_path):
            solver.load(best_model_path)
            logger.info(f"Loaded best model from epoch {best_epoch}")
        else:
            logger.warning(f"Best model checkpoint not found: {best_model_path}")

    return solver, best_epoch

def test(cfg, solver, logger):
    logger.info("="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80)

    valid_metrics = solver.evaluate("valid")
    test_metrics = solver.evaluate("test")

    logger.info("Validation Metrics:")
    for k, v in valid_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info("Test Metrics:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    total_train_samples = sum(len(ts) for ts in solver.train_sets)
    total_params = sum(p.numel() for p in solver.models.shared_model.parameters())
    trainable_params = sum(p.numel() for p in solver.models.shared_model.parameters() if p.requires_grad)

    logger.info(f"Total datasets: {len(solver.train_sets)}, Total training samples: {total_train_samples:,}")
    logger.info(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,} "
                f"({trainable_params/total_params*100:.2f}%)")

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

def create_working_directory(cfg):
    output_dir = os.path.expanduser(cfg.output_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if "--phase2" in sys.argv or "--full" in sys.argv:
        output_dir = os.path.join(output_dir, f"phase2_baseline_{timestamp}")
    else:
        output_dir = os.path.join(output_dir, f"baseline_multitask_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    return output_dir

if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)

    cfg = load_config(args.config)

    cfg.train.num_epoch = 4
    cfg.train.batch_size = 8
    cfg.train.gradient_interval = 6
    cfg.optimizer.lr = 3e-5
    cfg.engine.batch_size = 8
    cfg.engine.log_interval = 50

    phase_name = "Full Baseline"
    print(f"STARTING {phase_name}: {cfg.train.num_epoch} epochs, batch_size={cfg.train.batch_size}")

    if args.resume or args.resume_auto:
        output_dir = os.getcwd()
        print(f"Resuming training in existing directory: {output_dir}")
    else:
        output_dir = setup_drive_output(cfg)
        print(f"Created output directory: {output_dir}")

    logger = get_root_logger()
    logger.info(f"{phase_name} training started")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Output directory: {output_dir}")

    if not (args.resume or args.resume_auto):
        with open("config_used.yaml", "w") as f:
            yaml.dump(dict(cfg), f, default_flow_style=False)
        logger.info("Configuration saved")

    try:
        logger.info("Building solver with full datasets...")
        solver = build_solver(cfg, logger, limit_samples=None)

        total_samples = sum(len(ts) for ts in solver.train_sets)
        est_time_per_epoch = (total_samples // (cfg.train.batch_size * cfg.train.gradient_interval)) * 0.4 / 60
        est_total_time = est_time_per_epoch * cfg.train.num_epoch
        logger.info(f"Estimated total training time: {est_total_time:.1f} hours")
        logger.info(f"Colab 24h limit: {'SAFE' if est_total_time < 20 else 'RISKY'}")

        logger.info("Starting FULL BASELINE training...")
        solver, best_epoch = train_and_validate(cfg, solver, logger)
        logger.info(f"Training completed. Best epoch: {best_epoch}")

        test(cfg, solver, logger)
        logger.info(f"{phase_name} training and evaluation completed successfully!")

        if torch.cuda.is_available():
            logger.info(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        with open('error_log.txt', 'w') as f:
            f.write(f"Error: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        raise
