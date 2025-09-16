import os
import sys
import warnings
warnings.filterwarnings('ignore')
from datasets import load_dataset
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

def check_requirements():
    required_packages = [
        'torch', 'transformers', 'datasets', 'tqdm', 
        'numpy', 'pandas', 'yaml', 'easydict'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + ' '.join(missing))
        return False
    
    print("All required packages are installed")
    return True

def test_proteinglm_datasets():
    print("\nTesting proteinglm datasets...")
    
    datasets_to_test = [
        "proteinglm/ssp_q8",
        "proteinglm/stability_prediction",
        "proteinglm/peptide_HLA_MHC_affinity"
    ]
    
    working_datasets = []
    for dataset_name in datasets_to_test:
        try:
            print(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)
            print(f"[DEBUG] First example from {dataset_name}:")
            print(dataset['train'][0])
            print(f"{dataset_name} - OK")
            working_datasets.append(dataset_name)
        except Exception as e:
            print(f"{dataset_name} - Failed: {str(e)}")
    
    return working_datasets

def create_minimal_config(working_datasets):
    dataset_mapping = {
        "proteinglm/stability_prediction": "Thermostability",
        "proteinglm/ssp_q8": "SecondaryStructure",
        "proteinglm/peptide_HLA_MHC_affinity": "PeptideHLAMHCAffinity"
    }
    
    config = {
        'output_dir': './quick_start_outputs',
        
        'model': {
            'type': 'shared_lora',  # Use shared backbone with LoRA
            'model_name': 'Rostlab/prot_bert_bfd',
            'readout': 'pooler',
            'freeze_bert': True,
            'lora_rank': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1
        },
        
        'datasets': [],
        'tasks': [],
        
        'train': {
            'num_epoch': 2,
            'batch_size': 4,
            'gradient_interval': 2,
            'tradeoff': 0.5
        },
        
        'optimizer': {
            'type': 'AdamW',
            'lr': 2e-5,
            'weight_decay': 0.01
        },
        
        'engine': {
            'batch_size': 4,
            'num_worker': 0,
            'log_interval': 10
        },
        
        'eval_metric': 'accuracy'
    }
    
    for i, dataset_name in enumerate(working_datasets[:3]):
        dataset_type = dataset_mapping[dataset_name]
        is_center = (i == 0)
        
        config['datasets'].append({
            'type': dataset_type,
            'path': './data',
            'center': is_center
        })
        
        if dataset_type == 'SecondaryStructure':
            config['tasks'].append({
                'type': 'token_classification',
                'num_labels': 8,
                'loss': 'cross_entropy'
            })
        elif dataset_type == 'PeptideHLAMHCAffinity':
            config['tasks'].append({
                'type': 'classification',
                'num_labels': 2,
                'loss': 'cross_entropy'
            })
        elif dataset_type == 'Thermostability':
            config['tasks'].append({
                'type': 'regression',
                'num_labels': 1,
                'loss': 'mse'
            })
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return config

def run_quick_training_with_shared_backbone(config):
    """
    Updated training function using shared backbone architecture
    """
    print("\nStarting quick training with shared backbone...")
    
    try:
        from flip_hf import Thermostability, SecondaryStructure, PeptideHLAMHCAffinity
        from engine_hf import create_shared_multitask_model, SharedBackboneModelsWrapper
        from engine_hf import MultiTaskEngine
        from torch.utils.data import DataLoader
        
        print("Loading datasets...")
        train_sets, valid_sets, test_sets = [], [], []
        
        for dataset_config in config['datasets']:
            config_copy = dataset_config.copy()
            dataset_type = config_copy.pop('type')
            is_center = config_copy.pop('center', False)
            
            if dataset_type == 'Thermostability':
                dataset = Thermostability(**config_copy)
            elif dataset_type == 'SecondaryStructure':
                dataset = SecondaryStructure(**config_copy)
            elif dataset_type == 'PeptideHLAMHCAffinity':
                dataset = PeptideHLAMHCAffinity(**config_copy)
            
            train_set, valid_set, test_set = dataset.split()
            
            # Use small subsets for quick testing
            train_subset = torch.utils.data.Subset(train_set, range(min(100, len(train_set))))
            valid_subset = torch.utils.data.Subset(valid_set, range(min(50, len(valid_set))))
            test_subset = torch.utils.data.Subset(test_set, range(min(50, len(test_set))))
            
            if is_center:
                train_sets = [train_subset] + train_sets
                valid_sets = [valid_subset] + valid_sets
                test_sets = [test_subset] + test_sets
            else:
                train_sets.append(train_subset)
                valid_sets.append(valid_subset)
                test_sets.append(test_subset)
        
        print(f"Loaded {len(train_sets)} datasets")
        
        print("Creating shared backbone model...")
        # Create single shared model instead of multiple separate models
        shared_model = create_shared_multitask_model(
            tasks_config=config['tasks'],
            model_config=config['model']
        )
        
        # Print model info
        total_params = sum(p.numel() for p in shared_model.parameters())
        trainable_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create optimizer for shared model
        optimizer = AdamW(shared_model.parameters(), lr=config['optimizer']['lr'])
        
        # Create wrapper for compatibility with existing engine
        task_names = [f"Task_{i}" for i in range(len(config['tasks']))]
        models_wrapper = SharedBackboneModelsWrapper(shared_model, task_names)
        
        # Modified MultiTaskEngine to work with shared backbone
        class SharedBackboneMultiTaskEngine(MultiTaskEngine):
            def __init__(self, shared_model, train_sets, valid_sets, test_sets, optimizer, **kwargs):
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
                self.scheduler = None
                
                self.epoch = 0
                self.step = 0
        
        # Initialize engine with shared backbone
        engine = SharedBackboneMultiTaskEngine(
            shared_model=shared_model,
            train_sets=train_sets,
            valid_sets=valid_sets,
            test_sets=test_sets,
            optimizer=optimizer,
            batch_size=config['engine']['batch_size'],
            num_worker=config['engine']['num_worker']
        )
        
        # Debug: Check batching
        debug_loader = DataLoader(
            train_sets[0],
            batch_size=config['train']['batch_size'],
            collate_fn=engine.collate_fn
        )
        first_batch = next(iter(debug_loader))
        print("\nDEBUG: First batch structure:")
        print("Keys:", first_batch.keys())
        if "targets" in first_batch:
            print("Targets type:", type(first_batch["targets"]))
            if isinstance(first_batch["targets"], dict):
                for k, v in first_batch["targets"].items():
                    print(f"  - {k}: {type(v)}, shape={getattr(v, 'shape', None)}")
        
        print("Training with shared backbone...")
        engine.train(num_epoch=config['train']['num_epoch'], tradeoff=config['train']['tradeoff'])
        
        print("Evaluating...")
        metrics = engine.evaluate("valid")
        
        print("✅ Quick training with shared backbone completed successfully!")
        print("Results:", metrics)
        
        # Show memory usage comparison
        print(f"\nMemory Usage Comparison:")
        print(f"Shared backbone: {total_params:,} parameters")
        print(f"Separate models would have: {total_params * len(config['tasks']):,} parameters")
        print(f"Memory savings: {((total_params * len(config['tasks']) - total_params) / (total_params * len(config['tasks']))) * 100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("ProteinGLM Multi-Task Learning - Quick Start (Shared Backbone)")
    print("=" * 70)
    
    if not check_requirements():
        return

    working_datasets = test_proteinglm_datasets()
    
    if len(working_datasets) == 0:
        print("\nNo proteinglm datasets are working!")
        return
    elif len(working_datasets) == 1:
        print(f"\nOnly 1 dataset working: {working_datasets[0]}")
        print("Multi-task learning needs at least 2 datasets.")
        return
    else:
        print(f"\n{len(working_datasets)} datasets working: {working_datasets}")
    
    config = create_minimal_config(working_datasets)
    
    os.makedirs('quick_start_outputs', exist_ok=True)
    with open('quick_start_config_shared.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nConfig saved to: quick_start_config_shared.yaml")
    
    success = run_quick_training_with_shared_backbone(config)
    
    if success:
        print("\nSUCCESS! Shared backbone multi-task setup is working!")
        print("\nKey Benefits of Shared Backbone:")
        print("1. Memory efficient - single backbone for all tasks")
        print("2. True multi-task learning - tasks share representations")
        print("3. Better generalization - backbone learns from all tasks")
        print("\nNext steps:")
        print("1. Modify config for your full experiment")
        print("2. Experiment with different task combinations")
        print("3. Try different LoRA configurations")
    else:
        print("\nSomething went wrong. Check the error messages above.")

if __name__ == "__main__":
    main()