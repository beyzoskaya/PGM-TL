import os
import sys
import yaml
import torch
import logging
from easydict import EasyDict
from torch.utils.data import Subset

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from engine_hf import MultiTaskEngine, create_shared_multitask_model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config_hf.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase", type=str, default="single_task")
    return parser.parse_known_args()[0]

def set_seed(seed):
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(log_file="single_task_training.log"):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def load_config(config_file="config_hf.yaml"):
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return EasyDict(yaml.safe_load(f))
    else:
        cfg = {
            'output_dir': './single_task_outputs',
            'model': {
                'type': 'shared_lora',
                'model_name': 'Rostlab/prot_bert_bfd',
                'readout': 'pooler',
                'freeze_bert': True,
                'lora_rank': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1
            },
            'train': {'num_epoch': 4, 'batch_size': 8, 'gradient_interval': 1},
            'optimizer': {'type': 'AdamW', 'lr': 3e-5, 'weight_decay': 0.01},
            'scheduler': {'type': 'StepLR', 'step_size': 3, 'gamma': 0.5},
            'engine': {'batch_size': 8, 'num_worker': 1, 'log_interval': 50},
            'eval_metric': 'accuracy',
        }
        return EasyDict(cfg)

def create_single_task_dataset(task_type, path="./data", limit_samples=None):
    if task_type == 'SecondaryStructure':
        dataset = SecondaryStructure(path=path)
    elif task_type == 'Thermostability':
        dataset = Thermostability(path=path)
    elif task_type == 'CloningCLF':
        dataset = CloningCLF(path=path)
    else:
        raise ValueError(f"Unknown task: {task_type}")

    train_set, valid_set, test_set = dataset.split()
    if limit_samples:
        train_set = Subset(train_set, range(min(len(train_set), limit_samples.get('train', len(train_set)))))
        valid_set = Subset(valid_set, range(min(len(valid_set), limit_samples.get('valid', len(valid_set)))))
        test_set = Subset(test_set, range(min(len(test_set), limit_samples.get('test', len(test_set)))))
    return train_set, valid_set, test_set


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger()
    cfg = load_config(args.config)
    cfg.optimizer.lr = float(cfg.optimizer.lr)
    cfg.optimizer.weight_decay = float(cfg.optimizer.weight_decay)

    tasks = [
        {"name": "SecondaryStructure", "type": "token_classification", "num_labels": 8, "loss": "cross_entropy"},
        {"name": "Thermostability", "type": "regression", "num_labels": 1, "loss": "mse"},
        {"name": "CloningCLF", "type": "binary_classification", "num_labels": 1, "loss": "binary_cross_entropy"}
    ]

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.chdir(cfg.output_dir)

    results = {}

    for task_cfg in tasks:
        task_name = task_cfg["name"]
        logger.info(f"\n===== Training single task: {task_name} =====")

        train_set, valid_set, test_set = create_single_task_dataset(task_name)
        logger.info(f"Train samples: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")

        # Create model for this single task
        model = create_shared_multitask_model(
            tasks_config=[task_cfg],
            model_config=cfg.model
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        scheduler = None  # optional: StepLR(optimizer, step_size=3, gamma=0.5)

        # Initialize MultiTaskEngine with a single model in a list
        solver = MultiTaskEngine(
            tasks=[model],  
            train_sets=[train_set],
            valid_sets=[valid_set],
            test_sets=[test_set],
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=cfg.engine.batch_size,
            num_worker=cfg.engine.num_worker,
            gradient_interval=1,
            log_interval=cfg.engine.log_interval
        )

        solver, best_epoch = solver.train(num_epoch=cfg.train.num_epoch)
        logger.info(f"Completed training for {task_name}, best epoch: {best_epoch}")

        metrics_valid = solver.evaluate("valid")
        metrics_test = solver.evaluate("test")
        logger.info(f"Validation metrics: {metrics_valid}")
        logger.info(f"Test metrics: {metrics_test}")

        results[task_name] = {
            "valid": metrics_valid,
            "test": metrics_test,
            "best_epoch": best_epoch
        }

        solver.save(f"{task_name}_best_model.pth")

    import json
    with open("single_task_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Single-task training completed for all tasks!")
