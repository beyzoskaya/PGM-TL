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
from engine_hf import MultiTaskEngine
from engine_hf import create_shared_multitask_model, SharedBackboneModelsWrapper
from collections import defaultdict

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
    config = {
        'output_dir': './outputs',
     
        'model': {
            'type': 'shared_lora',  # Use shared backbone with LoRA
            'model_name': 'Rostlab/prot_bert_bfd',
            'readout': 'pooler',
            'freeze_bert': True,
            'lora_rank': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'dropout': 0.1
        },
        
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
                'type': 'token_classification',  # Fixed: secondary structure is token classification
                'num_labels': 8,  # 8-class secondary structure
                'loss': 'cross_entropy'
            }
        ],
        
        # Training configuration
        'train': {
            'num_epoch': 1,  # Single epoch for quick testing
            'batch_size': 4,  # Small batch size for testing
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
            'batch_size': 4,  # Small batch size for testing
            'num_worker': 0,  # No multiprocessing for debugging
            'log_interval': 10
        },
        
        # Evaluation
        'eval_metric': 'accuracy',
        'test_batch_size': 8
    }
    
    return EasyDict(config)


def load_config(config_file):
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return EasyDict(config)
    else:
        print(f"Config file {config_file} not found. Using default configuration.")
        return create_default_config()


def create_datasets(dataset_configs):
    train_sets, valid_sets, test_sets = [], [], []
    
    for dataset_config in dataset_configs:
        config_copy = dataset_config.copy()
        dataset_type = config_copy.pop('type')
        is_center = config_copy.pop('center', False)
        
        # Create dataset
        if dataset_type == 'Thermostability':
            dataset = Thermostability(**config_copy)
        elif dataset_type == 'SecondaryStructure':
            dataset = SecondaryStructure(**config_copy)
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


def create_optimizer(shared_model, optimizer_config):
    """Create optimizer for shared model"""
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
    """Build the multi-task learning solver with shared backbone"""
    
    # Create datasets
    train_sets, valid_sets, test_sets = create_datasets(cfg.datasets)
    
    # Create shared multi-task model
    shared_model = create_shared_multitask_model(cfg.tasks, cfg.model)
    
    # Print model info
    total_params = sum(p.numel() for p in shared_model.parameters())
    trainable_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
    logger.info(f"Shared model - Total parameters: {total_params:,}")
    logger.info(f"Shared model - Trainable parameters: {trainable_params:,}")
    
    # Create optimizer for shared model
    optimizer = create_optimizer(shared_model, cfg.optimizer)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, cfg.get('scheduler'))
    
    # Create wrapper for compatibility with existing engine
    task_names = [f"Task_{i}" for i in range(len(cfg.tasks))]
    models_wrapper = SharedBackboneModelsWrapper(shared_model, task_names)
    
    class SharedBackboneMultiTaskEngine(MultiTaskEngine):
        def __init__(self, shared_model, train_sets, valid_sets, test_sets, optimizer, scheduler=None, **kwargs):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.batch_size = kwargs.get('batch_size', 4)
            self.gradient_interval = kwargs.get('gradient_interval', 1)
            self.num_worker = kwargs.get('num_worker', 0)
            self.log_interval = kwargs.get('log_interval', 100)
            
            # Use the shared model wrapper
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
    
    return solver

def train_and_validate(cfg, solver, logger):
    step = math.ceil(cfg.train.num_epoch / 10)
    best_score = float("-inf")
    best_epoch = -1

    if not cfg.train.num_epoch > 0:
        return solver, best_epoch

    for i in range(0, cfg.train.num_epoch, step):
        # Training
        num_epoch = min(step, cfg.train.num_epoch - i)
        logger.info(f"Training for {num_epoch} epoch(s)...")
        
        solver.train(
            num_epoch=num_epoch,
            tradeoff=cfg.train.get('tradeoff', 1.0)
        )
        
        # Save checkpoint
        checkpoint_path = f"model_epoch_{solver.epoch}.pth"
        solver.save(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Validation
        logger.info("Running validation...")
        metrics = solver.evaluate("valid")
        
        # Compute score for early stopping
        score = []
        for k, v in metrics.items():
            if cfg.eval_metric in k.lower():
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
        logger.info(f"Loading best model from epoch {best_epoch}")
        solver.load(f"model_epoch_{best_epoch}.pth")
    
    return solver, best_epoch


def test(cfg, solver, logger):
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
    output_dir = os.path.expanduser(cfg.output_dir)
    
    # Create unique directory with timestamp
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"shared_multitask_{timestamp}")
    
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
        print("Note: Contrastive learning not yet implemented with shared backbone")
    
    # Create working directory
    output_dir = create_working_directory(cfg)
    
    # Setup logging
    logger = get_root_logger()
    
    logger.info("Starting SHARED BACKBONE multi-task protein language model training")
    logger.info(f"Configuration: {pprint.pformat(dict(cfg))}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Save configuration
    with open("config_used.yaml", "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)
    
    try:
        # Build solver
        logger.info("Building shared backbone solver...")
        solver = build_solver(cfg, logger)
        
        # Train and validate
        logger.info("Starting training with shared backbone...")
        solver, best_epoch = train_and_validate(cfg, solver, logger)
        logger.info(f"Training completed. Best epoch: {best_epoch}")
        
        # Final testing
        test(cfg, solver, logger)
        
        logger.info("SHARED BACKBONE training and evaluation completed successfully!")
        
        # Memory usage summary
        if torch.cuda.is_available():
            logger.info(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
        
        logger.info("Benefits achieved:")
        logger.info("✓ Single shared ProtBert backbone for all tasks")
        logger.info("✓ Efficient memory usage")
        logger.info("✓ True multi-task representation learning")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise