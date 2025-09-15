import os
import sys
import warnings
warnings.filterwarnings('ignore')

"""
Creates dataset objects from flip_hf.py and 
models from protbert_hf.py then creates a 
MultiTaskEngine from engine_hf.py and 
calls engine.train()
"""

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
        print(f" Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + ' '.join(missing))
        return False
    
    print("All required packages are installed")
    return True


def test_proteinglm_datasets():
    print("\nTesting proteinglm datasets...")
    
    datasets_to_test = [
        "proteinglm/ssp_q8", # Secondary Structure (multi-class classification)
        "proteinglm/stability_prediction",  # Thermostability (regression)
        "proteinglm/peptide_HLA_MHC_affinity" # Peptide-HLA-MHC Affinity (binary classification)
    ]
    
    working_datasets = []
    for dataset_name in datasets_to_test:
        try:
            from datasets import load_dataset
            print(f"Testing {dataset_name}...")
            dataset = load_dataset(dataset_name)
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
            'type': 'lora',
            'model_name': 'Rostlab/prot_bert_bfd',
            'readout': 'pooler',
            'freeze_bert': True,
            'lora_rank': 8,  # Smaller for faster training
            'lora_alpha': 16,
            'lora_dropout': 0.1
        },
        
        'datasets': [],
        'tasks': [],
        
        'train': {
            'num_epoch': 2,  # Quick test
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
    
    for i, dataset_name in enumerate(working_datasets[:3]):  # Max 3 datasets
        dataset_type = dataset_mapping[dataset_name]
        is_center = (i == 0)  # First one is center task
        
        config['datasets'].append({
            'type': dataset_type,
            'path': './data',
            'center': is_center
        })
        
        # Add corresponding task
        if dataset_type == 'SecondaryStructure':
            config['tasks'].append({
                'type': 'token_classification',
                'num_labels': 8,
                'loss': 'cross_entropy'
            })
        elif dataset_type == 'PeptideHLAMHCAffinity':
            config['tasks'].append({
                'type': 'classification',  # multi-class classification with 2 labels
                'num_labels': 2,
                'loss': 'cross_entropy'
            })
        else:  # Regression tasks
            config['tasks'].append({
                'type': 'regression', 
                'num_labels': 1,
                'loss': 'mse'
            })
    
    return config

def run_quick_training(config):
    print("\n Starting quick training...")
    
    try:
        from flip_hf import Thermostability, SecondaryStructure, PeptideHLAMHCAffinity
        from protbert_hf import create_protbert_model
        from engine_hf import MultiTaskEngine
        
        import torch
        from torch.optim import AdamW
        
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
        
        print("Creating models...")
        tasks = []
        for task_config in config['tasks']:
            model = create_protbert_model(
                model_type='lora',
                model_name=config['model']['model_name'],
                num_labels=task_config['num_labels'],
                readout=config['model']['readout'],
                lora_rank=config['model']['lora_rank'],
                lora_alpha=config['model']['lora_alpha'],
                task_type=task_config['type'] 
            )
            tasks.append(model)
        
        all_params = []
        for task in tasks:
            all_params.extend(list(task.parameters()))
        
        optimizer = AdamW(all_params, lr=config['optimizer']['lr'])
        
        engine = MultiTaskEngine(
            tasks=tasks,
            train_sets=train_sets,
            valid_sets=valid_sets,
            test_sets=test_sets,
            optimizer=optimizer,
            scheduler=None,
            batch_size=config['engine']['batch_size'],
            num_worker=config['engine']['num_worker']
        )
        
        print("Training...")
        engine.train(num_epoch=config['train']['num_epoch'], tradeoff=config['train']['tradeoff'])
        
        print("Evaluating...")
        metrics = engine.evaluate("valid")
        
        print("✅ Quick training completed successfully!")
        print("Results:", metrics)
        
        return True
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("ProteinGLM Multi-Task Learning - Quick Start")
    print("=" * 60)
    
    if not check_requirements():
        return

    working_datasets = test_proteinglm_datasets()
    
    if len(working_datasets) == 0:
        print("\nNo proteinglm datasets are working!")
        print("Possible solutions:")
        print("1. Check if you have access to the proteinglm organization")
        print("2. Verify the dataset names are correct")
        print("3. Try alternative datasets (run test_datasets.py)")
        return
    elif len(working_datasets) == 1:
        print(f"\nOnly 1 dataset working: {working_datasets[0]}")
        print("Multi-task learning needs at least 2 datasets.")
        print("This will run as single-task learning.")
    else:
        print(f"\n{len(working_datasets)} datasets working: {working_datasets}")
    
    config = create_minimal_config(working_datasets)
    
    import yaml
    os.makedirs('quick_start_outputs', exist_ok=True)
    with open('quick_start_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nConfig saved to: quick_start_config.yaml")
    
    success = run_quick_training(config)
    
    if success:
        print("\nSUCCESS! setup is working!")
        print("Next steps:")
        print("1. Modify config_hf.yaml for your full experiment")  
        print("2. Run: python main_hf.py --config config_hf.yaml")
        print("3. Add contrastive learning: --use_contrastive")
    else:
        print("\nSomething went wrong. Check the error messages above.")

if __name__ == "__main__":
    main()