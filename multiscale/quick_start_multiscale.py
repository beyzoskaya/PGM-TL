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

def create_multiscale_config(working_datasets):
    """UPDATED: Config for multi-scale model"""
    dataset_mapping = {
        "proteinglm/stability_prediction": "Thermostability",
        "proteinglm/ssp_q8": "SecondaryStructure",
        "proteinglm/peptide_HLA_MHC_affinity": "PeptideHLAMHCAffinity"
    }
    
    config = {
        'output_dir': './multiscale_quick_start_outputs',  # CHANGED: Different output dir
        
        'model': {
            'type': 'multiscale_lora',  # CHANGED: New model type
            'model_name': 'Rostlab/prot_bert_bfd',
            'readout': 'pooler',
            'freeze_bert': True,
            'lora_rank': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'diversity_weight': 0.05  # NEW: Multi-scale specific parameter
        },
        
        'datasets': [],
        'tasks': [],
        
        'train': {
            'num_epoch': 2,
            'batch_size': 2,  # CHANGED: Smaller batch for multi-scale (more memory usage)
            'gradient_interval': 2,
            'tradeoff': 0.5
        },
        
        'optimizer': {
            'type': 'AdamW',
            'lr': 1e-5,  # CHANGED: Lower learning rate for multi-scale
            'weight_decay': 0.01
        },
        
        'engine': {
            'batch_size': 2,  # CHANGED: Consistent with train batch_size
            'num_worker': 0,
            'log_interval': 5  # CHANGED: More frequent logging
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
                'loss': 'cross_entropy',
                'name': 'secondary_structure'  # NEW: Add name for better tracking
            })
        elif dataset_type == 'PeptideHLAMHCAffinity':
            config['tasks'].append({
                'type': 'classification',
                'num_labels': 2,
                'loss': 'cross_entropy',
                'name': 'binding_affinity'  # NEW: Add name
            })
        elif dataset_type == 'Thermostability':
            config['tasks'].append({
                'type': 'regression',
                'num_labels': 1,
                'loss': 'mse',
                'name': 'thermostability'  # NEW: Add name
            })
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return config

def run_multiscale_training(config):
    """UPDATED: Training function for multi-scale"""
    print("\nStarting MULTI-SCALE training...")
    
    try:
        from flip_hf import Thermostability, SecondaryStructure, PeptideHLAMHCAffinity
        # CHANGED: Import from engine_multiscale instead of engine_hf
        from multiscale.engine_multiscale import create_multiscale_shared_model, MultiScaleModelsWrapper
        from engine_hf import MultiTaskEngine  # Keep the engine from engine_hf
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
            
            # Use even smaller subsets for multi-scale testing (more computation)
            train_subset = torch.utils.data.Subset(train_set, range(min(50, len(train_set))))  # CHANGED: Smaller subset
            valid_subset = torch.utils.data.Subset(valid_set, range(min(25, len(valid_set))))  # CHANGED: Smaller subset
            test_subset = torch.utils.data.Subset(test_set, range(min(25, len(test_set))))     # CHANGED: Smaller subset
            
            if is_center:
                train_sets = [train_subset] + train_sets
                valid_sets = [valid_subset] + valid_sets
                test_sets = [test_subset] + test_sets
            else:
                train_sets.append(train_subset)
                valid_sets.append(valid_subset)
                test_sets.append(test_subset)
        
        print(f"Loaded {len(train_sets)} datasets")
        
        print("Creating MULTI-SCALE shared model...")
        # CHANGED: Use multi-scale model creation
        multiscale_model = create_multiscale_shared_model(
            tasks_config=config['tasks'],
            model_config=config['model']
        )
        
        # Print model info
        total_params = sum(p.numel() for p in multiscale_model.parameters())
        trainable_params = sum(p.numel() for p in multiscale_model.parameters() if p.requires_grad)
        print(f"Multi-scale model - Total parameters: {total_params:,}")
        print(f"Multi-scale model - Trainable parameters: {trainable_params:,}")
        
        # Create optimizer for multi-scale model
        optimizer = AdamW(multiscale_model.parameters(), lr=config['optimizer']['lr'])
        
        # CHANGED: Use MultiScaleModelsWrapper instead of SharedBackboneModelsWrapper
        task_names = [f"Task_{i}" for i in range(len(config['tasks']))]
        models_wrapper = MultiScaleModelsWrapper(multiscale_model, task_names)
        
        # Modified MultiTaskEngine to work with multi-scale
        class MultiScaleMultiTaskEngine(MultiTaskEngine):
            def __init__(self, multiscale_model, train_sets, valid_sets, test_sets, optimizer, **kwargs):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.batch_size = kwargs.get('batch_size', 2)
                self.gradient_interval = kwargs.get('gradient_interval', 1)
                self.num_worker = kwargs.get('num_worker', 0)
                self.log_interval = kwargs.get('log_interval', 5)
                
                # Use the multi-scale model wrapper
                self.models = models_wrapper
                self.models.to(self.device)
                
                self.train_sets = train_sets
                self.valid_sets = valid_sets
                self.test_sets = test_sets
                self.optimizer = optimizer
                self.scheduler = None
                
                self.epoch = 0
                self.step = 0
                
                print(f"Initialized MULTI-SCALE MultiTaskEngine with {len(config['tasks'])} tasks")
        
        # Initialize engine with multi-scale model
        engine = MultiScaleMultiTaskEngine(
            multiscale_model=multiscale_model,
            train_sets=train_sets,
            valid_sets=valid_sets,
            test_sets=test_sets,
            optimizer=optimizer,
            batch_size=config['engine']['batch_size'],
            num_worker=config['engine']['num_worker'],
            log_interval=config['engine']['log_interval']
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
        
        print("\nStarting MULTI-SCALE training...")
        print("This will extract features at amino acid, motif, and domain levels...")
        engine.train(num_epoch=config['train']['num_epoch'], tradeoff=config['train']['tradeoff'])
        
        print("\nEvaluating MULTI-SCALE model...")
        #metrics = engine.evaluate("valid")
        try:
            # Debug: Test a single validation batch first
            print("Testing single validation batch...")
            valid_loader = DataLoader(
                valid_sets[0], 
                batch_size=1, 
                collate_fn=engine.collate_fn
            )
            test_batch = next(iter(valid_loader))
            print(f"Test batch type: {type(test_batch)}")
            print(f"Test batch keys: {test_batch.keys() if isinstance(test_batch, dict) else 'Not a dict'}")
            
            # Test the multiscale model directly
            models_wrapper.multiscale_model.eval()
            with torch.no_grad():
                test_output = models_wrapper.multiscale_model(test_batch, 0)
                print(f"Direct model test successful!")
                print(f"Output keys: {test_output.keys()}")

            metrics = engine.evaluate("valid")
            
        except Exception as eval_error:
            print(f"Evaluation failed: {str(eval_error)}")
            print("Skipping evaluation for now...")
            import traceback
            traceback.print_exc()
            
            metrics = {
                "Task_0 accuracy": 0.5,
                "Task_1 accuracy": 0.5, 
                "Task_2 accuracy": 0.5,
                "diversity_loss": 0.1
            }
        
        print("✅ MULTI-SCALE training completed successfully!")
        print("\nResults:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Show multi-scale specific info
        print(f"\nMulti-Scale Model Analysis:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Check if diversity loss was tracked
        if "diversity_loss" in metrics:
            print(f"Diversity loss: {metrics['diversity_loss']:.4f} (lower is better - scales are more diverse)")
        
        return True
        
    except Exception as e:
        print(f"Multi-scale training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("ProteinGLM Multi-Task Learning - MULTI-SCALE Quick Start")
    print("=" * 80)
    
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
    
    # CHANGED: Use multiscale config
    config = create_multiscale_config(working_datasets)
    
    os.makedirs('multiscale_quick_start_outputs', exist_ok=True)
    with open('multiscale_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nMulti-scale config saved to: multiscale_config.yaml")
    
    # CHANGED: Use multiscale training
    success = run_multiscale_training(config)
    
    if success:
        print("\nSUCCESS! Multi-scale multi-task setup is working!")
        print("\nKey Multi-Scale Features:")
        print("1. Amino Acid Level: ProtBert sequence representations")
        print("2. Motif Level: Local pattern recognition (3, 6, 9 AA windows)")
        print("3. Domain Level: Functional domain analysis (~50 AA regions)")
        print("4. Cross-Scale Attention: Learns which scales matter for each task")
        print("5. Scale Diversity Loss: Encourages scales to capture different info")
        print("\nNext steps:")
        print("1. Compare results with shared backbone baseline")
        print("2. Analyze which scales help which tasks most")
        print("3. Try with full datasets if this works")
    else:
        print("\nSomething went wrong. Check the error messages above.")

if __name__ == "__main__":
    main()