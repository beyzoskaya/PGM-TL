import os
import yaml
import torch
import logging
from easydict import EasyDict
from torch.utils.data import Subset
from flip_hf import CloningCLF
from engine_hf import MultiTaskEngine, create_shared_multitask_model
import json

def set_seed(seed=42):
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def load_config():
    cfg = {
        'output_dir': '/content/drive/MyDrive/protein_multitask_outputs/single_task_outputs',
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
        'eval_metric': 'accuracy'
    }
    return EasyDict(cfg)

def create_dataset(path="./data", limit_samples=None):
    dataset = CloningCLF(path=path)
   
    train_set, valid_set, test_set = dataset.split()

    if limit_samples:
        train_set = Subset(train_set, range(min(len(train_set), limit_samples.get('train', len(train_set)))))
        valid_set = Subset(valid_set, range(min(len(valid_set), limit_samples.get('valid', len(valid_set)))))
        test_set = Subset(test_set, range(min(len(test_set), limit_samples.get('test', len(test_set)))))

    return train_set, valid_set, test_set


class SingleTaskWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, batch):
        return self.model(batch, task_id=0)

if __name__ == "__main__":
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except:
        pass
    
    set_seed(42)
    cfg = load_config()

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_file = os.path.join(cfg.output_dir, "single_task_cloningclf.log")
    logger = get_logger(log_file)

    task_cfg = {
        "name": "CloningCLF",
        "type": "binary_classification",
        "num_labels": 1,
        "loss": "binary_cross_entropy"
    }

    logger.info("===== Training Single Task: CloningCLF =====")
    train_set, valid_set, test_set = create_dataset()

    shared_model = create_shared_multitask_model(
        tasks_config=[task_cfg],
        model_config=cfg.model
    )

    wrapped_model = SingleTaskWrapper(shared_model)

    optimizer = torch.optim.AdamW(
        wrapped_model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )

    solver = MultiTaskEngine(
        tasks=[wrapped_model],
        train_sets=[train_set],
        valid_sets=[valid_set],
        test_sets=[test_set],
        optimizer=optimizer,
        scheduler=None,
        batch_size=cfg.engine.batch_size,
        num_worker=cfg.engine.num_worker,
        gradient_interval=cfg.train.gradient_interval,
        log_interval=cfg.engine.log_interval
    )

    solver.train(num_epoch=cfg.train.num_epoch)
    best_epoch = solver.epoch
    logger.info(f"Training complete at epoch {best_epoch}")

    metrics_valid = solver.evaluate("valid")
    metrics_test = solver.evaluate("test")

    logger.info(f"Validation metrics: {metrics_valid}")
    logger.info(f"Test metrics: {metrics_test}")

    model_path = os.path.join(cfg.output_dir, "CloningCLF_best_model.pth")
    solver.save(model_path)
    logger.info(f"Model saved to {model_path}")

    results = {
        "CloningCLF": {
            "valid": metrics_valid,
            "test": metrics_test,
            "best_epoch": best_epoch
        }
    }
    with open(os.path.join(cfg.output_dir, "single_task_cloningclf_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("CloningCLF single-task training completed successfully!")

