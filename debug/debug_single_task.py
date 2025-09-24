# debug_single_task.py
import torch
from flip_hf import Thermostability
from protbert_hf import create_protbert_model
from torch.optim import AdamW

print("Testing single task...")

# Test 1: Thermostability (regression) - should be easiest
try:
    print("Loading Thermostability dataset...")
    dataset = Thermostability(path='./data', verbose=1)
    train_set, valid_set, test_set = dataset.split()
    
    # Take small subset
    small_train = torch.utils.data.Subset(train_set, range(10))
    
    print("Creating regression model...")
    model = create_protbert_model(
        model_type='lora',
        model_name='Rostlab/prot_bert_bfd',
        num_labels=1,  # Regression
        readout='pooler',
        lora_rank=8,
        lora_alpha=16,
        task_type='classification'  # Use classification even for regression
    )
    
    print("Testing single batch...")
    from torch.utils.data import DataLoader
    from engine_hf import MultiTaskEngine
    
    # Create simple dataloader
    engine = MultiTaskEngine(
        tasks=[model],
        train_sets=[small_train],
        valid_sets=[small_train],
        test_sets=[small_train],
        optimizer=AdamW(model.parameters(), lr=1e-5),
        batch_size=2,
        num_worker=0
    )
    
    # Get one batch and test
    dataloader = DataLoader(small_train, batch_size=2, collate_fn=engine.collate_fn)
    batch = next(iter(dataloader))
    batch = engine.move_to_device(batch)
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Sequences: {len(batch['sequence'])}")
    print(f"Targets keys: {batch['targets'].keys()}")
    print(f"Target values type: {type(batch['targets']['label'])}")
    print(f"Target values: {batch['targets']['label']}")
    
    # Test model forward
    print("Testing model forward...")
    outputs = model(batch)
    print(f"Model output keys: {outputs.keys()}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Test loss computation manually
    print("Testing loss computation...")
    loss = engine.models.compute_default_loss(outputs, batch)
    print(f"Loss: {loss}")
    
    print("✅ Thermostability works!")
    
except Exception as e:
    print(f"❌ Thermostability failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)

# Test 2: Secondary Structure (token classification) 
try:
    print("Loading SecondaryStructure dataset...")
    from flip_hf import SecondaryStructure
    dataset = SecondaryStructure(path='./data', verbose=1)
    train_set, valid_set, test_set = dataset.split()
    
    # Take small subset
    small_train = torch.utils.data.Subset(train_set, range(5))
    
    print("Creating token classification model...")
    model = create_protbert_model(
        model_type='lora',
        model_name='Rostlab/prot_bert_bfd',
        num_labels=8,  # 8 classes
        readout='mean',  # Try mean instead of pooler for token tasks
        lora_rank=8,
        lora_alpha=16,
        task_type='classification'
    )
    
    # Test single batch
    dataloader = DataLoader(small_train, batch_size=1, collate_fn=engine.collate_fn)
    batch = next(iter(dataloader))
    batch = engine.move_to_device(batch)
    
    print(f"Batch targets: {batch['targets']}")
    print(f"Target type: {type(batch['targets']['label'])}")
    print(f"Target sample: {batch['targets']['label'][0][:10] if len(batch['targets']['label']) > 0 else 'Empty'}")
    
    # Test model forward
    outputs = model(batch)
    print(f"Logits shape: {outputs['logits'].shape}")
    
    print("✅ SecondaryStructure works!")
    
except Exception as e:
    print(f"❌ SecondaryStructure failed: {e}")
    import traceback
    traceback.print_exc()