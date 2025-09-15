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

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from flip_hf import create_dataset, Thermostability, SecondaryStructure
from protbert_hf import create_protbert_model
from engine_hf import MultiTaskEngine, MultiTaskWithContrastive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="config_hf.yaml")
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--use_contrastive", help="use contrastive loss", action="store_true")
    parser.add_argument("--contrastive_weight", help="contrastive loss weight", type=float, default=0.1)

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
    """Create a default configuration for HuggingFace-based training"""
    config = {
        'output_dir': './outputs',
        
        # Model configuration
        'model': {
            'type': 'lora',  # base, classification, token_classification, lora
            'model_name': 'Rostlab/prot_bert_bfd',
            'readout': 'pooler',
            'freeze_bert': True,
            'lora_rank': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'dropout': 0.1
        },
        
        # Dataset configurations
        'datasets': [
            {
                'type': 'Thermostability',
                'path': './data',
                'split': 'human_cell',
                'center': True  # This will be the center task
            },
            {
                'type': 'SecondaryStructure', 
                'path': './data',
                'center': False  # Auxiliary task
            }
        ],
        
        # Task configurations
        'tasks': [
            {
                'type': 'regression',
                'num_labels': 1,
                'loss': 'mse'
            },
            {
                'type': 'classification',
                'num_labels': 8,  # 8-class secondary structure
                'loss': 'cross_entropy'
            }
        ],
        
        # Training configuration
        'train': {
            'num_epoch': 10,
            'batch_size': 8,
            'gradient_interval': 1,
            'tradeoff': 0.5  # Weight for auxiliary tasks
        },
        
        # Optimizer configuration
        'optimizer': {
            'type': 'AdamW',
            'lr': 2e-5,
            'weight_decay': 0.01
        },
        
        # Scheduler configuration (optional)
        'scheduler': {
            'type': 'StepLR',
            'step_size': 3,
            'gamma': 0.5
        },
        
        # Engine configuration
        'engine': {
            'batch_size': 8,
            'num_worker': 2,
            'log_interval': 100
        },
        
        # Evaluation
        'eval_metric': 'accuracy',
        'test_batch_size': 16
    }
    
    return EasyDict(config)


def load_config(config_file):
    """Load configuration from YAML file"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return EasyDict(config)
    else:
        print(f"Config file {config_file} not found. Using default configuration.")
        return create_default_config()


def create_datasets(dataset_configs):
    """Create datasets from configuration"""
    train_sets, valid_sets, test_sets = [], [], []
    
    for dataset_config in dataset_configs:
        dataset_type = dataset_config.pop('type')
        is_center = dataset_config.pop('center', False)
        
        # Create dataset
        if dataset_type == 'Thermostability':
            dataset = Thermostability(**dataset_config)
        elif dataset_type == 'SecondaryStructure':
            dataset = SecondaryStructure(**dataset_config)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Split dataset
        train_set, valid_set, test_set = dataset.split()
        
        print(f"Dataset {dataset_type}: Train={len(train_set)}, Valid={len(valid_set)}, Test={len(test_set)}")
        
        # Add to lists (center task goes first)
        if is_center:
            train_sets = [train_set] + train_sets
            valid_sets = [valid_set] + valid_sets
            test_sets = [test_set] + test_sets
        else:
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)
    
    return train_sets, valid_sets, test_sets


def create_tasks(task_configs, model_config):
    """Create task models from configuration"""
    tasks = []
    
    for i, task_config in enumerate(task_configs):
        task_type = task_config.get('type', 'classification')
        num_labels = task_config.get('num_labels', 1)
        
        # Create model based on task type
        if model_config['type'] == 'lora':
            # LoRA model
            task_model = create_protbert_model(
                model_type='lora',
                model_name=model_config['model_name'],
                num_labels=num_labels,
                readout=model_config['readout'],
                lora_rank=model_config['lora_rank'],
                lora_alpha=model_config['lora_alpha'],
                lora_dropout=model_config['lora_dropout'],
                task_type='classification' if task_type == 'classification' else 'classification'
            )
        else:
            # Regular model
            if task_type == 'classification':
                task_model = create_protbert_model(
                    model_type='classification',
                    model_name=model_config['model_name'],
                    num_labels=num_labels,
                    readout=model_config['readout'],
                    freeze_bert=model_config.get('freeze_bert', False),
                    dropout=model_config.get('dropout', 0.1)
                )
            else:
                # Regression task - use classification model with 1 output
                task_model = create_protbert_model(
                    model_type='classification',
                    model_name=model_config['model_name'],
                    num_labels=1,
                    readout=model_config['readout'],
                    freeze_bert=model_config.get('freeze_bert', False),
                    dropout=model_config.get('dropout', 0.1)
                )
        
        tasks.append(task_model)
    
    return tasks


def create_optimizer(tasks, optimizer_config):
    """Create optimizer for all task models"""
    # Collect all parameters
    all_params = []
    for task in tasks:
        all_params.extend(list(task.parameters()))
    
    optimizer_type = optimizer_config.get('type', 'AdamW')
    lr = optimizer_config.get('lr', 2e-5)
    weight_decay = optimizer_config.get('weight_decay', 0.01)
    
    if optimizer_type == 'AdamW':
        optimizer = AdamW(all_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = Adam(all_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, scheduler_config):
    """Create learning rate scheduler"""
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


def build_solver(cfg, logger):
    """Build the multi-task learning solver"""
    
    # Create datasets
    train_sets, valid_sets, test_sets = create_datasets(cfg.datasets)
    
    # Create task models
    tasks = create_tasks(cfg.tasks, cfg.model)
    
    # Create optimizer
    optimizer = create_optimizer(tasks, cfg.optimizer)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, cfg.get('scheduler'))
    
    # Create engine
    engine_kwargs = cfg.engine.copy()
    
    if cfg.get('use_contrastive', False):
        # Use contrastive learning engine
        solver = MultiTaskWithContrastive(
            tasks=tasks,
            train_sets=train_sets,
            valid_sets=valid_sets,
            test_sets=test_sets,
            optimizer=optimizer,
            scheduler=scheduler,
            contrastive_weight=cfg.get('contrastive_weight', 0.1),
            **engine_kwargs
        )
    else:
        # Standard multi-task engine
        solver = MultiTaskEngine(
            tasks=tasks,
            train_sets=train_sets,
            valid_sets=valid_sets,
            test_sets=test_sets,
            optimizer=optimizer,
            scheduler=scheduler,
            **engine_kwargs
        )
    
    return solver


def train_and_validate(cfg, solver, logger):
    """Train and validate the model"""
    step = math.ceil(cfg.train.num_epoch / 10)
    best_score = float("-inf")
    best_epoch = -1

    if not cfg.train.num_epoch > 0:
        return solver, best_epoch

    for i in range(0, cfg.train.num_epoch, step):
        # Training
        num_epoch = min(step, cfg.train.num_epoch - i)
        solver.train(
            num_epoch=num_epoch,
            tradeoff=cfg.train.get('tradeoff', 1.0)
        )
        
        # Save checkpoint
        solver.save(f"model_epoch_{solver.epoch}.pth")
        
        # Validation
        metrics = solver.evaluate("valid")
        
        # Compute score for early stopping
        score = []
        for k, v in metrics.items():
            if k.startswith(cfg.eval_metric):
                if "mse" in cfg.eval_metric or "rmse" in cfg.eval_metric:
                    score.append(-v)  # Lower is better for MSE/RMSE
                else:
                    score.append(v)  # Higher is better for accuracy
        
        if score:
            current_score = sum(score) / len(score)
            if current_score > best_score:
                best_score = current_score
                best_epoch = solver.epoch
                logger.info(f"New best score: {best_score:.4f} at epoch {best_epoch}")

    # Load best model
    if best_epoch > 0:
        solver.load(f"model_epoch_{best_epoch}.pth")
    
    return solver, best_epoch


def test(cfg, solver, logger):
    """Test the final model"""
    logger.info("Running final evaluation...")
    
    # Validation set
    valid_metrics = solver.evaluate("valid")
    logger.info("Validation Results:")
    for k, v in valid_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Test set
    test_metrics = solver.evaluate("test")
    logger.info("Test Results:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")


def get_root_logger():
    """Setup logging"""
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler("training.log")
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def create_working_directory(cfg):
    """Create output directory"""
    output_dir = os.path.expanduser(cfg.output_dir)
    
    # Create unique directory with timestamp
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"multitask_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    return output_dir


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Update config with command line arguments
    if args.use_contrastive:
        cfg.use_contrastive = True
        cfg.contrastive_weight = args.contrastive_weight
    
    # Create working directory
    output_dir = create_working_directory(cfg)
    
    # Setup logging
    logger = get_root_logger()
    
    logger.info("Starting multi-task protein language model training")
    logger.info(f"Configuration: {pprint.pformat(dict(cfg))}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Save configuration
    with open("config_used.yaml", "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)
    
    try:
        # Build solver
        logger.info("Building solver...")
        solver = build_solver(cfg, logger)
        
        # Train and validate
        logger.info("Starting training...")
        solver, best_epoch = train_and_validate(cfg, solver, logger)
        logger.info(f"Training completed. Best epoch: {best_epoch}")
        
        # Final testing
        test(cfg, solver, logger)
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise