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

from flip_hf import create_dataset, Thermostability, SecondaryStructure, PeptideHLAMHCAffinity
from engine_hf import MultiTaskEngine
from engine_hf import create_shared_multitask_model, SharedBackboneModelsWrapper
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", default="config_hf.yaml")
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--use_contrastive", help="use contrastive loss", action="store_true")
    parser.add_argument("--contrastive_weight", help="contrastive loss weight", type=float, default=0.1)
    parser.add_argument("--phase1", help="quick validation with limited data", action="store_true")
    parser.add_argument("--phase2", help="full baseline training", action="store_true")
    parser.add_argument("--full", help="full baseline training", action="store_true")
    parser.add_argument("--limit_train", type=int, default=None, help="Limit training samples per dataset")
    parser.add_argument("--limit_valid", type=int, default=None, help="Limit validation samples per dataset")
    parser.add_argument("--limit_test", type=int, default=None, help="Limit test samples per dataset")
   
    parser.add_argument("--resume", help="path to checkpoint to resume from", type=str, default=None)
    parser.add_argument("--resume_auto", help="auto-resume from latest checkpoint in output dir", action="store_true")

    return parser.parse_known_args()[0]

def set_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
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
            {
                'type': 'SecondaryStructure',    # CENTER TASK (Task 0)
                'path': './data',
                'center': True
            },
            {
                'type': 'Thermostability',       # AUX TASK (Task 1)
                'path': './data',
                'split': 'human_cell',
                'center': False
            },
            {
                'type': 'PeptideHLAMHCAffinity', # AUX TASK (Task 2)
                'path': './data',
                'center': False
            }
        ],
        
        'tasks': [
            {
                'type': 'token_classification',  # Task 0: SecondaryStructure
                'num_labels': 8,
                'loss': 'cross_entropy'
            },
            {
                'type': 'regression',            # Task 1: Thermostability
                'num_labels': 1,
                'loss': 'mse'
            },
            {
                'type': 'binary_classification', # Task 2: PeptideHLA
                'num_labels': 1,
                'loss': 'cross_entropy'
            }
        ],
        
        'train': {
            'num_epoch': 6,
            'batch_size': 8,
            'gradient_interval': 4,
            'tradeoff': 0.5
        },
        
        'optimizer': {
            'type': 'AdamW',
            'lr': 2e-5,
            'weight_decay': 0.01
        },
        
        'scheduler': {
            'type': 'StepLR',
            'step_size': 3,
            'gamma': 0.5
        },
        
        'engine': {
            'batch_size': 8,
            'num_worker': 1,
            'log_interval': 50
        },
        
        'eval_metric': 'accuracy',
        'test_batch_size': 16
    }
    
    return EasyDict(config)

def setup_drive_output(cfg):
    import time
    
    if not os.path.exists('/content/drive/MyDrive'):
        raise RuntimeError("Google Drive not mounted. Please run 'drive.mount('/content/drive')' in a Colab cell first.")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_drive_path = "/content/drive/MyDrive/protein_multitask_outputs"
    
    import sys
    if "--full" in sys.argv:
        output_dir = f"{base_drive_path}/full_baseline_{timestamp}"
    else:
        output_dir = f"{base_drive_path}/baseline_multitask_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    print(f"Output directory (in Drive): {output_dir}")
    return output_dir

def load_config(config_file):
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return EasyDict(config)
    else:
        print(f"Config file {config_file} not found. Using default configuration.")
        return create_default_config()

def create_datasets_limit_option(dataset_configs, limit_samples=None):

    train_sets, valid_sets, test_sets = [], [], []
    
    print("=" * 60)
    print("LOADING & CONFIGURING DATASETS")
    print("=" * 60)
    
    for i, dataset_config in enumerate(dataset_configs):
        config_copy = dataset_config.copy()
        dataset_type = config_copy.pop('type')
        is_center = config_copy.pop('center', False)
        
        print(f"\n[Dataset {i}] Loading {dataset_type}...")
        print(f"  Center task: {is_center}")
        
        # Create dataset
        if dataset_type == 'Thermostability':
            dataset = Thermostability(**config_copy)
        elif dataset_type == 'SecondaryStructure':
            dataset = SecondaryStructure(**config_copy)
        elif dataset_type == 'PeptideHLAMHCAffinity':
            dataset = PeptideHLAMHCAffinity(**config_copy)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        train_set, valid_set, test_set = dataset.split()
        original_sizes = (len(train_set), len(valid_set), len(test_set))
        print(f"  Original sizes - Train: {original_sizes[0]}, Valid: {original_sizes[1]}, Test: {original_sizes[2]}")
        
        if limit_samples and not is_center:  
            train_limit = limit_samples.get('train', len(train_set))
            valid_limit = limit_samples.get('valid', len(valid_set))
            test_limit = limit_samples.get('test', len(test_set))
            
            if len(train_set) > train_limit:
                train_indices = torch.randperm(len(train_set))[:train_limit]
                train_set = Subset(train_set, train_indices)
            if len(valid_set) > valid_limit:
                valid_indices = torch.randperm(len(valid_set))[:valid_limit]
                valid_set = Subset(valid_set, valid_indices)
            if len(test_set) > test_limit:
                test_indices = torch.randperm(len(test_set))[:test_limit]
                test_set = Subset(test_set, test_indices)
            
            print(f"  Limited sizes - Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")
        elif is_center:
            print(f"  Using FULL center task dataset")
        
        if is_center:
            train_sets = [train_set] + train_sets
            valid_sets = [valid_set] + valid_sets
            test_sets = [test_set] + test_sets
            print("  Added as CENTER task")
        else:
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)
            print("  Added as auxiliary task")
    
    total_train = sum(len(ts) for ts in train_sets)
    total_valid = sum(len(vs) for vs in valid_sets)
    total_test = sum(len(ts) for ts in test_sets)
    
    print(f"\n[SUMMARY] Total samples - Train: {total_train}, Valid: {total_valid}, Test: {total_test}")
    
    return train_sets, valid_sets, test_sets

def create_datasets(dataset_configs, limit_samples=None):
    train_sets, valid_sets, test_sets = [], [], []
    
    print("=" * 60)
    print("LOADING FULL DATASETS (NO LIMITING)")
    print("=" * 60)
    
    for i, dataset_config in enumerate(dataset_configs):
        config_copy = dataset_config.copy()
        dataset_type = config_copy.pop('type')
        is_center = config_copy.pop('center', False)
        
        print(f"\n[Dataset {i}] Loading {dataset_type}...")
        print(f"  Center task: {is_center}")
        
        if dataset_type == 'Thermostability':
            dataset = Thermostability(**config_copy)
        elif dataset_type == 'SecondaryStructure':
            dataset = SecondaryStructure(**config_copy)
        elif dataset_type == 'PeptideHLAMHCAffinity':
            dataset = PeptideHLAMHCAffinity(**config_copy)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        train_set, valid_set, test_set = dataset.split()
        print(f"  Using FULL dataset - Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")
        
        if is_center:
            train_sets = [train_set] + train_sets
            valid_sets = [valid_set] + valid_sets
            test_sets = [test_set] + test_sets
            print("  Added as CENTER task")
        else:
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)
            print("  Added as auxiliary task")
    
    total_train = sum(len(ts) for ts in train_sets)
    total_valid = sum(len(vs) for vs in valid_sets)
    total_test = sum(len(ts) for ts in test_sets)
    
    print(f"\n[SUMMARY] FULL DATASETS - Train: {total_train}, Valid: {total_valid}, Test: {total_test}")
    
    return train_sets, valid_sets, test_sets

def create_optimizer(shared_model, optimizer_config):

    optimizer_type = optimizer_config.get('type', 'AdamW')
    lr = optimizer_config.get('lr', 2e-5)
    weight_decay = optimizer_config.get('weight_decay', 0.01)
    
    if optimizer_type == 'AdamW':
        optimizer = AdamW(shared_model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = Adam(shared_model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, scheduler_config):

    if scheduler_config is None:
        return None
    
    scheduler_type = scheduler_config.get('type', 'StepLR')
    
    if scheduler_type == 'StepLR':
        step_size = scheduler_config.get('step_size', 3)
        gamma = scheduler_config.get('gamma', 0.5)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'CosineAnnealingLR':
        T_max = scheduler_config.get('T_max', 10)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler

def build_solver(cfg, logger, limit_samples=None):

    train_sets, valid_sets, test_sets = create_datasets(cfg.datasets, limit_samples)
    
    print("\n" + "=" * 60)
    print("CREATING SHARED BACKBONE MODEL")
    print("=" * 60)
    
    print("Model configuration:")
    for key, value in cfg.model.items():
        print(f"  {key}: {value}")
    
    print("\nTask configurations:")
    for i, task in enumerate(cfg.tasks):
        print(f"  Task {i}: {task}")
    
    shared_model = create_shared_multitask_model(cfg.tasks, cfg.model)
    
    total_params = sum(p.numel() for p in shared_model.parameters())
    trainable_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Analysis:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    print(f"\nTask Heads:")
    for task_name, head in shared_model.task_heads.items():
        head_params = sum(p.numel() for p in head.parameters())
        print(f"  {task_name}: {head_params:,} parameters")
    
    logger.info(f"Shared model - Total parameters: {total_params:,}")
    logger.info(f"Shared model - Trainable parameters: {trainable_params:,}")
    
    optimizer = create_optimizer(shared_model, cfg.optimizer)
    print(f"\nOptimizer: {optimizer.__class__.__name__} with lr={cfg.optimizer.get('lr', 2e-5)}")

    scheduler = create_scheduler(optimizer, cfg.get('scheduler'))
    if scheduler:
        print(f"Scheduler: {scheduler.__class__.__name__}")
    
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
    
    print(f"\nDevice: {solver.device}")
    print(f"Batch size: {solver.batch_size}")
    print(f"Gradient interval: {solver.gradient_interval}")
    print(f"Effective batch size: {solver.batch_size * solver.gradient_interval}")
    
    return solver

def find_latest_checkpoint(output_dir):
    checkpoint_pattern = os.path.join(output_dir, "baseline_model_epoch_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None, 0

    epochs = []
    for cp in checkpoints:
        try:
            epoch_num = int(cp.split('_epoch_')[1].split('.pth')[0])
            epochs.append((epoch_num, cp))
        except:
            continue
    
    if not epochs:
        return None, 0
    
    latest_epoch, latest_checkpoint = max(epochs, key=lambda x: x[0])
    return latest_checkpoint, latest_epoch

def train_and_validate(cfg, solver, logger):
    step = 1  
    best_score = float("-inf")
    best_epoch = -1

    args = parse_args() 
    start_epoch = 0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        if os.path.exists(args.resume):
            solver.load(args.resume, load_optimizer=True)
            start_epoch = solver.epoch
            logger.info(f"Resumed from epoch {start_epoch}")
        else:
            logger.error(f"Resume checkpoint not found: {args.resume}")
            logger.info("Starting fresh training")
    elif args.resume_auto:
        checkpoint_path, start_epoch = find_latest_checkpoint(".")
        if checkpoint_path:
            logger.info(f"Auto-resuming from checkpoint: {checkpoint_path}")
            solver.load(checkpoint_path, load_optimizer=True)
            start_epoch = solver.epoch
            logger.info(f"Resumed from epoch {start_epoch}")
        else:
            logger.info("No checkpoint found for auto-resume, starting fresh")

    if cfg.train.num_epoch <= start_epoch:
        logger.info("Training already completed or no epochs remaining")
        return solver, best_epoch

    remaining_epochs = cfg.train.num_epoch - start_epoch
    total_train_samples = sum(len(ts) for ts in solver.train_sets)
    batches_per_epoch = total_train_samples // (solver.batch_size * solver.gradient_interval)
    estimated_time_per_epoch = batches_per_epoch * 0.5 / 60  # Rough estimate: 0.5 sec per batch
    
    logger.info("=" * 80)
    logger.info("TRAINING PHASE STARTED")
    logger.info("=" * 80)
    if start_epoch > 0:
        logger.info(f"RESUMING from epoch: {start_epoch}")
    logger.info(f"Target total epochs: {cfg.train.num_epoch}")
    logger.info(f"Remaining epochs: {remaining_epochs}")
    logger.info(f"Total training samples: {total_train_samples:,}")
    logger.info(f"Effective batch size: {solver.batch_size * solver.gradient_interval}")
    logger.info(f"Batches per epoch: ~{batches_per_epoch:,}")
    logger.info(f"Estimated time per epoch: ~{estimated_time_per_epoch:.1f} minutes")
    logger.info(f"Estimated remaining time: ~{estimated_time_per_epoch * remaining_epochs:.1f} minutes")
    logger.info(f"Tradeoff weight for auxiliary tasks: {cfg.train.get('tradeoff', 1.0)}")
    logger.info("=" * 80)

    for i in range(start_epoch, cfg.train.num_epoch, step):
        # Training
        num_epoch = min(step, cfg.train.num_epoch - i)
        logger.info(f"\nTraining epochs {i+1}-{i+num_epoch} of {cfg.train.num_epoch}...")
        
        solver.train(
            num_epoch=num_epoch,
            tradeoff=cfg.train.get('tradeoff', 1.0)
        )
        
        checkpoint_path = f"baseline_model_epoch_{solver.epoch}.pth"
        solver.save(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        latest_path = "latest_checkpoint.pth"
        solver.save(latest_path)
        
        progress_info = {
            'epoch': solver.epoch,
            'total_epochs': cfg.train.num_epoch,
            'completed': solver.epoch >= cfg.train.num_epoch
        }
        with open('training_progress.txt', 'w') as f:
            f.write(f"Completed epoch: {solver.epoch}/{cfg.train.num_epoch}\n")
            f.write(f"Last checkpoint: {checkpoint_path}\n")
            f.write(f"Training completed: {'Yes' if progress_info['completed'] else 'No'}\n")
        
        logger.info("Running validation...")
        metrics = solver.evaluate("valid")
        
        logger.info("=" * 60)
        logger.info(f"VALIDATION RESULTS - Epoch {solver.epoch}")
        logger.info("=" * 60)
        for k, v in metrics.items():
            if "Task_0" in k or "Center" in k:
                logger.info(f"  [CENTER] {k}: {v:.4f}")
            elif "Task_1" in k:
                logger.info(f"  [AUX-1] {k}: {v:.4f}")
            elif "Task_2" in k:
                logger.info(f"  [AUX-2] {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 60)

        score = []
        for k, v in metrics.items():
            if ("Task_0" in k or "Center" in k) and cfg.eval_metric in k.lower():
                if "mse" in cfg.eval_metric or "rmse" in cfg.eval_metric:
                    score.append(-v)  # Lower is better for MSE/RMSE
                else:
                    score.append(v)  # Higher is better for accuracy
        
        if score:
            current_score = sum(score) / len(score)
            if current_score > best_score:
                best_score = current_score
                best_epoch = solver.epoch
                logger.info(f"NEW BEST CENTER TASK SCORE: {best_score:.4f} at epoch {best_epoch}")

                best_model_path = f"best_baseline_model_epoch_{best_epoch}.pth"
                solver.save(best_model_path)
                logger.info(f"Saved best model: {best_model_path}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory: {memory_used:.1f}GB used, {memory_reserved:.1f}GB reserved")
        
        if solver.scheduler and hasattr(solver.optimizer, 'param_groups'):
            current_lr = solver.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.2e}")

        remaining = cfg.train.num_epoch - solver.epoch
        logger.info(f"Progress: {solver.epoch}/{cfg.train.num_epoch} epochs completed, {remaining} remaining")

    if best_epoch > 0:
        logger.info(f"\nLoading best model from epoch {best_epoch}")
        best_model_path = f"best_baseline_model_epoch_{best_epoch}.pth"
        if os.path.exists(best_model_path):
            solver.load(best_model_path)
        else:
            logger.warning(f"Best model checkpoint not found: {best_model_path}")

    with open('training_completed.txt', 'w') as f:
        f.write(f"Training completed successfully\n")
        f.write(f"Total epochs: {cfg.train.num_epoch}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best score: {best_score:.4f}\n")
    
    return solver, best_epoch

def test(cfg, solver, logger):
    logger.info("\n" + "=" * 80)
    logger.info("FINAL BASELINE EVALUATION")
    logger.info("=" * 80)
    
    # Validation set
    logger.info("Validation Set Results:")
    valid_metrics = solver.evaluate("valid")
    for k, v in valid_metrics.items():
        if "Task_0" in k or "Center" in k:
            logger.info(f"  [CENTER] {k}: {v:.4f}")
        elif "Task_1" in k:
            logger.info(f"  [AUX-1] {k}: {v:.4f}")
        elif "Task_2" in k:
            logger.info(f"  [AUX-2] {k}: {v:.4f}")
    
    logger.info("-" * 60)
    
    # Test set  
    logger.info("Test Set Results:")
    test_metrics = solver.evaluate("test")
    for k, v in test_metrics.items():
        if "Task_0" in k or "Center" in k:
            logger.info(f"  [CENTER] {k}: {v:.4f}")
        elif "Task_1" in k:
            logger.info(f"  [AUX-1] {k}: {v:.4f}")
        elif "Task_2" in k:
            logger.info(f"  [AUX-2] {k}: {v:.4f}")
    
    logger.info("=" * 80)
    logger.info("BASELINE SUMMARY")
    logger.info("=" * 80)
    
    total_train_samples = sum(len(ts) for ts in solver.train_sets)
    total_params = sum(p.numel() for p in solver.models.shared_model.parameters())
    trainable_params = sum(p.numel() for p in solver.models.shared_model.parameters() if p.requires_grad)
    
    logger.info(f"Total datasets: 3 ({len(solver.train_sets)} tasks)")
    logger.info(f"Total training samples: {total_train_samples:,}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"Memory savings vs separate models: ~66.7%")
    logger.info(f"Training completed: {cfg.train.num_epoch} epochs")
    
    # Task-specific summary
    logger.info("\nTask-specific Final Results:")
    for metric_name, value in test_metrics.items():
        if "Task_0" in metric_name or "Center" in metric_name:
            logger.info(f"  Secondary Structure (CENTER): {value:.4f}")
        elif "Task_1" in metric_name:
            logger.info(f"  Thermostability (AUX): {value:.4f}")
        elif "Task_2" in metric_name:
            logger.info(f"  HLA-MHC Affinity (AUX): {value:.4f}")
    
    logger.info("=" * 80)

def get_root_logger():
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler("baseline_training.log")
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
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
    
    phase_name = "Full Baseline"
    print("FULL BASELINE: Using all dataset instances")
    
    cfg.train.num_epoch = 6 
    cfg.train.batch_size = 12  
    cfg.train.gradient_interval = 4 
    
    cfg.engine.batch_size = 12
    cfg.engine.num_worker = 0  
    cfg.engine.log_interval = 100 
    
    print(f"Configuration: {cfg.train.num_epoch} epochs, batch_size={cfg.train.batch_size}")
    
    if args.resume or args.resume_auto:
        output_dir = os.getcwd()
        print(f"Resuming training in: {output_dir}")
        logger = get_root_logger()
        logger.info(f"RESUME: {phase_name} training")
    else:
        output_dir = setup_drive_output(cfg)
        logger = get_root_logger()
        logger.info(f"NEW FULL BASELINE training")
        logger.info(f"Drive output directory: {output_dir}")
    
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("Using FULL datasets - no limiting applied")
    
    if not (args.resume or args.resume_auto):
        with open("config_used.yaml", "w") as f:
            yaml.dump(dict(cfg), f, default_flow_style=False)
        logger.info("Configuration saved to Drive")
    
    try:
        logger.info("Building solver with full datasets...")
        solver = build_solver(cfg, logger, limit_samples=None)

        total_samples = sum(len(ts) for ts in solver.train_sets)
        est_time_per_epoch = (total_samples // (cfg.train.batch_size * cfg.train.gradient_interval)) * 0.4 / 60
        est_total_time = est_time_per_epoch * cfg.train.num_epoch
        
        logger.info(f"Estimated training time: {est_total_time:.1f} hours total")
        logger.info(f"Colab Pro 24h limit: {'SAFE' if est_total_time < 20 else 'RISKY'}")

        logger.info("Starting FULL BASELINE training...")
        solver, best_epoch = train_and_validate(cfg, solver, logger)
        logger.info(f"Training completed. Best epoch: {best_epoch}")
        
        test(cfg, solver, logger)
        
        logger.info("FULL BASELINE training completed successfully!")
        logger.info(f"All checkpoints saved to: {output_dir}")
        
        if torch.cuda.is_available():
            logger.info(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        with open('error_log.txt', 'w') as f:
            f.write(f"Error: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        raise