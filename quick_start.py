import os
import sys
import warnings
warnings.filterwarnings('ignore')
from datasets import load_dataset
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from engine_hf import SharedBackboneModelsWrapper, MultiTaskEngine

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
    
    print("[OK] All required packages are installed")
    return True

def test_proteinglm_datasets():
    print("\n" + "="*60)
    print("TESTING DATASET AVAILABILITY")
    print("="*60)
    
    datasets_to_test = [
        "proteinglm/ssp_q8",
        "proteinglm/stability_prediction", 
        "proteinglm/cloning_clf"
    ]
    
    working_datasets = []
    for i, dataset_name in enumerate(datasets_to_test, 1):
        try:
            print(f"\n[{i}/3] Loading: {dataset_name}")
            dataset = load_dataset(dataset_name)
         
            first_example = dataset['train'][0]
            print(f"  [SUCCESS] Sample keys: {list(first_example.keys())}")
            print(f"  [INFO] Train samples: {len(dataset['train'])}")
   
            if 'seq' in first_example:
                print(f"  [INFO] Sample sequence length: {len(first_example['seq'])}")
            if 'label' in first_example:
                label_type = type(first_example['label'])
                if isinstance(first_example['label'], str):
                    print(f"  [INFO] Sample label: {first_example['label'][:20]}... (string)")
                elif isinstance(first_example['label'], list):
                    print(f"  [INFO] Sample label: List of {len(first_example['label'])} items")
                else:
                    print(f"  [INFO] Sample label: {first_example['label']} ({label_type.__name__})")
            
            working_datasets.append(dataset_name)
            
        except Exception as e:
            print(f"  [FAILED] {str(e)}")
    
    print(f"\n[RESULT] {len(working_datasets)}/{len(datasets_to_test)} datasets working")
    return working_datasets

def create_minimal_config(working_datasets):
    dataset_mapping = {
        "proteinglm/stability_prediction": "Thermostability",
        "proteinglm/ssp_q8": "SecondaryStructure", 
        "proteinglm/cloning_clf": "CloningCLF"
    }
    
    config = {
        'output_dir': './quick_start_outputs',
        
        'model': {
            'type': 'shared_lora',
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
    
    print("\n" + "="*60)
    print("BUILDING TASK CONFIGURATION")
    print("="*60)
    
    for i, dataset_name in enumerate(working_datasets[:3]):
        dataset_type = dataset_mapping[dataset_name]
        is_center = (i == 0)
        
        print(f"\n[Task {i}] {dataset_type}")
        print(f"  Center task: {is_center}")
        print(f"  Dataset: {dataset_name}")
        
        config['datasets'].append({
            'type': dataset_type,
            'path': './data',
            'center': is_center
        })
        
        if dataset_type == 'SecondaryStructure':
            task_config = {
                'type': 'token_classification',
                'num_labels': 8,
                'loss': 'cross_entropy'
            }
            
        elif dataset_type == 'CloningCLF':
            task_config = {
                'type': 'binary_classification',  
                'num_labels': 1, 
                'loss': 'binary_cross_entropy'
            }
            
        elif dataset_type == 'Thermostability':
            task_config = {
                'type': 'regression',
                'num_labels': 1,
                'loss': 'mse'
            }
            
        config['tasks'].append(task_config)
    
    return config

def debug_dataset_loading(config):
    print("\n" + "="*60)
    print("LOADING & DEBUGGING DATASETS")
    print("="*60)
    
    train_sets, valid_sets, test_sets = [], [], []
    
    for i, dataset_config in enumerate(config['datasets']):
        config_copy = dataset_config.copy()
        dataset_type = config_copy.pop('type')
        is_center = config_copy.pop('center', False)
        
        print(f"\n[Dataset {i}] Loading {dataset_type}...")
        
        if dataset_type == 'Thermostability':
            dataset = Thermostability(**config_copy)
        elif dataset_type == 'SecondaryStructure':
            dataset = SecondaryStructure(**config_copy)
        elif dataset_type == 'CloningCLF':
            dataset = CloningCLF(**config_copy)
 
        train_set, valid_set, test_set = dataset.split()
        
        print(f"  Original sizes - Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")
        
        train_subset = torch.utils.data.Subset(train_set, range(min(100, len(train_set))))
        valid_subset = torch.utils.data.Subset(valid_set, range(min(50, len(valid_set))))
        test_subset = torch.utils.data.Subset(test_set, range(min(50, len(test_set))))
        
        print(f"  Debug sizes - Train: {len(train_subset)}, Valid: {len(valid_subset)}, Test: {len(test_subset)}")
        
        sample = train_subset[0]
        print(f"  Sample sequence length: {len(sample['sequence'])}")
        print(f"  Target keys: {list(sample['targets'].keys())}")
        
        for key, value in sample['targets'].items():
            if isinstance(value, list):
                print(f"    - {key}: List of {len(value)} items (first 3: {value[:3]})")
            else:
                print(f"    - {key}: {value} ({type(value).__name__})")
        
        if is_center:
            train_sets = [train_subset] + train_sets
            valid_sets = [valid_subset] + valid_sets
            test_sets = [test_subset] + test_sets
            print("  Added as CENTER task")
        else:
            train_sets.append(train_subset)
            valid_sets.append(valid_subset)
            test_sets.append(test_subset)
            print("  Added as auxiliary task")
    
    return train_sets, valid_sets, test_sets

def debug_model_creation(config):
    print("\n" + "="*60)
    print("CREATING SHARED BACKBONE MODEL")
    print("="*60)
    
    from engine_hf import create_shared_multitask_model
    
    print("Model configuration:")
    for key, value in config['model'].items():
        print(f"  {key}: {value}")
    
    print("\nTask configurations:")
    for i, task in enumerate(config['tasks']):
        print(f"  Task {i}: {task}")
    
    shared_model = create_shared_multitask_model(
        tasks_config=config['tasks'],
        model_config=config['model']
    )
    
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
        print(f"    Architecture: {head}")
    
    return shared_model

def debug_batch_processing(engine, train_sets):
    print("\n" + "="*60)
    print("DEBUGGING BATCH PROCESSING")
    print("="*60)
    
    for task_id, train_set in enumerate(train_sets):
        print(f"\n[Task {task_id}] Creating sample batch...")
        
        debug_loader = DataLoader(
            train_set,
            batch_size=2,  
            collate_fn=engine.collate_fn,
            shuffle=False
        )
        
        batch = next(iter(debug_loader))
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Sequences: {len(batch['sequence'])} items")
        print(f"    - Sample lengths: {[len(seq) for seq in batch['sequence']]}")
        
        print(f"  Targets structure:")
        if isinstance(batch['targets'], dict):
            for key, values in batch['targets'].items():
                if isinstance(values, torch.Tensor):
                    print(f"    - {key}: tensor shape {values.shape}, dtype {values.dtype}")
                    print(f"      Sample values: {values.flatten()[:5].tolist()}")
                elif isinstance(values, list):
                    if values and isinstance(values[0], list):
                        print(f"    - {key}: List of lists, lengths {[len(v) for v in values]}")
                        print(f"      First sequence: {values[0][:10]}...")
                    else:
                        print(f"    - {key}: List of {len(values)} items: {values}")
                else:
                    print(f"    - {key}: {type(values)} - {values}")
        
        print(f"  Testing forward pass...")
        try:
            device = next(engine.models.parameters()).device
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                else:
                    batch_device[key] = value
            
            task_model = engine.models[task_id]
            with torch.no_grad():
                outputs = task_model(batch_device)
            
            print(f"    [SUCCESS] Output keys: {list(outputs.keys())}")
            print(f"    Logits shape: {outputs['logits'].shape}")
            
            loss = engine.models.compute_default_loss(outputs, batch_device)
            print(f"    Loss: {loss.item():.4f}")
            
            #metrics = engine.models.compute_default_metrics(outputs, batch_device)
            task_model = engine.models[task_id]
            task_type = getattr(task_model, 'task_type', None)
            metrics = engine.models.compute_default_metrics(outputs, batch_device, task_type=task_type)
            print(f"    Metrics: {metrics}")
            if task_id == 2:  
                print(f"[TASK 2 CHECK] Should show accuracy, not mse: {list(metrics.keys())}")
            
        except Exception as e:
            print(f"    [FAILED] Forward pass failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def run_quick_training_with_debug(config):
    print("\n" + "="*60)
    print("STARTING TRAINING WITH DEBUGGING")
    print("="*60)
    
    try:
        train_sets, valid_sets, test_sets = debug_dataset_loading(config)
        
        shared_model = debug_model_creation(config)
        
        optimizer = AdamW(shared_model.parameters(), lr=config['optimizer']['lr'])
        print(f"\nOptimizer: {optimizer.__class__.__name__} with lr={config['optimizer']['lr']}")
        
        task_names = [f"Task_{i}" for i in range(len(config['tasks']))]
        models_wrapper = SharedBackboneModelsWrapper(shared_model, task_names)
        
        class DebugMultiTaskEngine(MultiTaskEngine):
            def __init__(self, shared_model, train_sets, valid_sets, test_sets, optimizer, **kwargs):
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
                self.scheduler = None
                
                self.epoch = 0
                self.step = 0
                
                print(f"Device: {self.device}")
                print(f"Batch size: {self.batch_size}")
        
        engine = DebugMultiTaskEngine(
            shared_model=shared_model,
            train_sets=train_sets,
            valid_sets=valid_sets,
            test_sets=test_sets,
            optimizer=optimizer,
            **config['engine']
        )
        
        batch_test_success = debug_batch_processing(engine, train_sets)
        if not batch_test_success:
            print("[CRITICAL] Batch processing test failed!")
            return False
        
        print("\n" + "="*60)
        print("TRAINING PHASE")
        print("="*60)
        
        print(f"Training for {config['train']['num_epoch']} epochs...")
        print(f"Tradeoff weight: {config['train']['tradeoff']}")
        
        engine.train(
            num_epoch=config['train']['num_epoch'], 
            tradeoff=config['train']['tradeoff']
        )
        
        print("\n" + "="*60)
        print("EVALUATION PHASE") 
        print("="*60)
        
        metrics = engine.evaluate("valid")
        
        print("\n[SUCCESS] TRAINING COMPLETED!")
        print("\nFinal Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        total_params = sum(p.numel() for p in shared_model.parameters())
        separate_model_params = total_params * len(config['tasks'])
        savings = ((separate_model_params - total_params) / separate_model_params) * 100
        
        print(f"\nMemory Efficiency:")
        print(f"  Shared backbone: {total_params:,} parameters")
        print(f"  Separate models would need: {separate_model_params:,} parameters")
        print(f"  Memory savings: {savings:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("ProteinGLM Multi-Task Learning - Baseline Debug")
    print("=" * 70)
    print("This debug version will verify each step before full training.")
    print("=" * 70)
    
    if not check_requirements():
        return
    
    working_datasets = test_proteinglm_datasets()
    
    if len(working_datasets) == 0:
        print("\n[CRITICAL] No datasets are working!")
        return
    elif len(working_datasets) == 1:
        print(f"\n[WARNING] Only 1 dataset working: {working_datasets[0]}")
        print("Multi-task learning needs at least 2 datasets.")
        return
    else:
        print(f"\n[OK] {len(working_datasets)} datasets ready for multi-task learning")
    
    config = create_minimal_config(working_datasets)
    
    os.makedirs('quick_start_outputs', exist_ok=True)
    with open('debug_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\nDebug config saved to: debug_config.yaml")
    
    success = run_quick_training_with_debug(config)
    
    if success:
        print("\n" + "="*70)
        print("BASELINE SETUP IS WORKING CORRECTLY")
        print("="*70)
        print("[OK] Dataset loading works")
        print("[OK] Model creation works") 
        print("[OK] Batch processing works")
        print("[OK] Training loop works")
        print("[OK] Evaluation works")
        
    else:
        print("\n" + "="*70)
        print("ISSUES DETECTED - REVIEW DEBUG OUTPUT ABOVE")
        print("="*70)

if __name__ == "__main__":
    main()

"""
Task 0 --> Secondary Structure --> Token classification (8 labels) --> accuracy
Task 1 --> Thermostability --> Regression (1 label) --> mse
Task 2 --> Peptide-HLA MHC Affinity --> Binary classification (1 label) --> accuracy
"""