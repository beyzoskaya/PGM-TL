"""
Test script to verify that the proteinglm datasets work correctly.
Run this before training to ensure all datasets load properly.
"""

from datasets import load_dataset
import traceback

def test_dataset_loading():
    """Test loading all three proteinglm datasets"""
    
    datasets_to_test = [
        "proteinglm/ssp_q8",
        "proteinglm/stability_prediction", 
        "proteinglm/peptide_HLA_MHC_affinity"
    ]
    
    results = {}
    
    for dataset_name in datasets_to_test:
        print(f"\n{'='*50}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Try to load the dataset
            dataset = load_dataset(dataset_name)
            
            print(f"✅ Successfully loaded {dataset_name}")
            print(f"Available splits: {list(dataset.keys())}")
            
            # Check first split for structure
            first_split = list(dataset.keys())[0]
            first_example = dataset[first_split][0]
            
            print(f"First example keys: {list(first_example.keys())}")
            print(f"Sample data:")
            for key, value in first_example.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
            
            print(f"Dataset size: {len(dataset[first_split])} examples")
            
            results[dataset_name] = {
                'status': 'success',
                'splits': list(dataset.keys()),
                'columns': list(first_example.keys()),
                'size': len(dataset[first_split])
            }
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}")
            print(f"Error: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            
            results[dataset_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results


def test_our_dataset_classes():
    
    print(f"\n{'='*60}")
    print("Testing our custom dataset classes")
    print(f"{'='*60}")

    try:
        from flip_hf import Thermostability, SecondaryStructure, PeptideHLAMHCAffinity
        
        datasets_to_test = [
            ('Thermostability', Thermostability),
            ('SecondaryStructure', SecondaryStructure),
            ('PeptideHLAMHCAffinity', PeptideHLAMHCAffinity)
        ]
        
        for name, dataset_class in datasets_to_test:
            print(f"\nTesting {name}...")
            
            try:
                dataset = dataset_class(path='./data', verbose=1)
                
                print(f"✅ {name} loaded successfully")
                print(f"Total sequences: {len(dataset.sequences)}")
                print(f"Split sizes: {dataset.num_samples}")
                print(f"Target keys: {list(dataset.targets.keys())}")
                
                # Test splits
                train_set, valid_set, test_set = dataset.split()
                print(f"Split lengths: Train={len(train_set)}, Valid={len(valid_set)}, Test={len(test_set)}")
                
                # Test getting an item
                if len(train_set) > 0:
                    sample = train_set[0]
                    print(f"Sample keys: {list(sample.keys())}")
                    if 'sequence' in sample:
                        seq_len = len(sample['sequence'])
                        print(f"Sample sequence length: {seq_len}")
                
            except Exception as e:
                print(f"❌ {name} failed: {str(e)}")
                # Don't print full traceback for cleaner output
    
    except ImportError as e:
        print(f"❌ Failed to import dataset classes: {e}")
        print("This means the flip_hf.py file needs to be updated.")


def check_data_compatibility():
    """Check if the data format is compatible with our training pipeline"""
    
    print(f"\n{'='*60}")
    print("Checking data compatibility")
    print(f"{'='*60}")
    
    try:
        from flip_hf import Thermostability
        
        # Load one dataset as example
        dataset = Thermostability(path='./data', verbose=0)
        train_set, _, _ = dataset.split()
        
        if len(train_set) > 0:
            sample = train_set[0]
            
            print("Sample structure:")
            for key, value in sample.items():
                print(f"  {key}: {type(value)}")
                if key == 'sequence':
                    print(f"    Length: {len(value)}")
                    print(f"    Sample: {value[:50]}...")
                elif key == 'targets':
                    print(f"    Target keys: {list(value.keys()) if isinstance(value, dict) else 'Not dict'}")
                
            # Test with tokenizer
            print("\nTesting with ProtBert tokenizer...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
            
            sequence = sample['sequence']
            spaced_seq = ' '.join(sequence)
            tokens = tokenizer(spaced_seq, return_tensors='pt', truncation=True, padding=True)
            
            print(f"✅ Tokenization successful")
            print(f"Input IDs shape: {tokens['input_ids'].shape}")
            print(f"Attention mask shape: {tokens['attention_mask'].shape}")
        
    except Exception as e:
        print(f"❌ Compatibility check failed: {e}")
        print("This is expected - we'll fix it in the next step.")


if __name__ == "__main__":
    print("🧪 Testing proteinglm datasets...")
    
    # Test 1: Raw dataset loading
    results = test_dataset_loading()
    
    # Test 2: Our dataset classes
    test_our_dataset_classes()
    
    # Test 3: Data compatibility
    check_data_compatibility()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    working_datasets = [name for name, result in results.items() if result['status'] == 'success']
    failed_datasets = [name for name, result in results.items() if result['status'] == 'failed']
    
    print(f"✅ Working datasets: {len(working_datasets)}")
    for dataset in working_datasets:
        print(f"  - {dataset}")
    
    print(f"❌ Failed datasets: {len(failed_datasets)}")
    for dataset in failed_datasets:
        print(f"  - {dataset}")
    
    if len(working_datasets) >= 2:
        print("\n🎉 You have enough datasets for multi-task learning!")
        print("You can proceed with training.")
    else:
        print("\n⚠️  You need at least 2 working datasets for multi-task learning.")
        print("Consider using alternative datasets or check your access permissions.")
    
    print(f"\nNext steps:")
    print(f"1. If datasets work: Run `python main_hf.py --config config_hf.yaml`")
    print(f"2. If datasets fail: Check dataset names and access permissions")
    print(f"3. Consider using alternative datasets from InstaDeepAI or biomap-research")