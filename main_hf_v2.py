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
from engine_hf import MultiTaskEngine, create_shared_multitask_model, SharedBackboneModelsWrapper

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

    print("="*60)
    print("LOADING DATASETS")
    print("="*60)

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

        sample = next(iter(train_set))
        if not isinstance(sample, dict) or 'input_ids' not in sample or 'labels' not in sample:
            raise ValueError(f"[{dtype}] dataset must return dict with 'input_ids' and 'labels' keys")

        input_shape = tuple(sample['input_ids'].shape)
        label_shape = tuple(sample['labels'].shape)
        input_shapes.append(input_shape)
        label_shapes.append(label_shape)

        print(f"[{dtype}] sample input shape: {input_shape}, label shape: {label_shape}")

        if is_center:
            train_sets = [train_set] + train_sets
            valid_sets = [valid_set] + valid_sets
            test_sets = [test_set] + test_sets
        else:
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)

        print(f"  Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")

    ref_input_shape = input_shapes[0]
    for idx, shape in enumerate(input_shapes):
        if shape[1:] != ref_input_shape[1:]:
            raise ValueError(
                f"\n[Shape Error] Input shape mismatch between datasets:\n"
                f"  {dataset_names[0]}: {ref_input_shape}\n"
                f"  {dataset_names[idx]}: {shape}\n"
                f"Ensure tokenizer and padding are consistent."
            )

    print("\n✅ All dataset input shapes are consistent with shared backbone.")

    for idx, shape in enumerate(label_shapes):
        print(f"Label shape for {dataset_names[idx]}: {shape}")

    total_train = sum(len(ts) for ts in train_sets)
    total_valid = sum(len(vs) for vs in valid_sets)
    total_test = sum(len(ts) for ts in test_sets)
    print(f"\nTotal samples - Train: {total_train}, Valid: {total_valid}, Test: {total_test}")

    print("\nPerforming batch consistency check...")

    temp_loader = DataLoader(train_sets[0], batch_size=2, shuffle=True)
    first_batch = next(iter(temp_loader))

    if isinstance(first_batch, dict):
        x = first_batch['input_ids']
        y = first_batch['labels']
    else:
        raise ValueError("Unexpected dataset format; expected dict from __getitem__")

    print(f"Batch check -> input_ids batch shape: {x.shape}, labels batch shape: {y.shape}")
    if x.ndim != 2 and x.ndim != 3:
        raise ValueError(f"Unexpected input_ids shape {x.shape}. Expected [batch, seq_len] or [batch, seq_len, hidden].")

    print("✅ Batch shape check passed.\n")

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
    shared_model = create_shared_multitask_model(cfg.tasks, cfg.model)

    print("===== Checking LoRA Layers =====")
    for name, param in shared_model.named_parameters():
        if "lora" in name.lower():
            print(name, param.shape, param.requires_grad)

    total_params = sum(p.numel() for p in shared_model.parameters())
    trainable_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    logger.info(f"Model Analysis: Total params={total_params}, Trainable={trainable_params} ({trainable_params/total_params*100:.1f}%), Frozen={frozen_params}")

    optimizer = create_optimizer(shared_model, cfg.optimizer)
    scheduler = create_scheduler(optimizer, cfg.get('scheduler'))

    task_names = [f"Task_{i}" for i in range(len(cfg.tasks))]
    models_wrapper = SharedBackboneModelsWrapper(shared_model, task_names)

    class SharedBackboneMultiTaskEngine(MultiTaskEngine):
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.batch_size = cfg.engine.batch_size
            self.gradient_interval = cfg.train.gradient_interval
            self.num_worker = cfg.engine.num_worker
            self.log_interval = cfg.engine.log_interval
            self.models = models_wrapper
            self.models.to(self.device)
            self.train_sets = train_sets
            self.valid_sets = valid_sets
            self.test_sets = test_sets
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.epoch = 0
            self.step = 0
            logger.info(f"Initialized MultiTaskEngine with {len(cfg.tasks)} tasks, device={self.device}")

    return SharedBackboneMultiTaskEngine()

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
        else:
            logger.info("No checkpoint found for auto-resume. Starting fresh.")

    solver, best_epoch = train_and_validate(cfg, solver, logger)
    test(cfg, solver, logger)
    logger.info(f"Training completed. Best epoch: {best_epoch}")

