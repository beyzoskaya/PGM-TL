import torch
import numpy as np

from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_semantic_prompt_tuning import multitask_collate_fn

def print_section(title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")

def inspect_pipeline():
    print_section("1. LOADING ACTUAL BACKBONE")

    print("Instantiating SharedProtBert...")
    backbone = SharedProtBert(lora_rank=1, unfrozen_layers=0)
 
    tokenizer = backbone.tokenizer
    print(f"✓ Tokenizer loaded from backbone: {type(tokenizer)}")
    
    print("\nLoading Datasets...")
    datasets = {
        'Thermostability': Thermostability(verbose=0),
        'SecondaryStructure': SecondaryStructure(verbose=0),
        'CloningCLF': CloningCLF(verbose=0)
    }

    print_section("2. DEEP DIVE: DATA PROCESSING STEPS")

    for task_name, ds in datasets.items():
        print(f"\n>>> INSPECTING TASK: {task_name} <<<")
        
        # A. RAW DATA (What comes out of flip_hf.py)
        raw_sample = ds[0]
        raw_seq = raw_sample['sequence']
        raw_target = raw_sample['targets']['target']
        
        print(f"  [A] Raw Data from Dataset:")
        print(f"      - Sequence (First 50): {raw_seq[:50]}...")
        print(f"      - Length: {len(raw_seq)}")
        print(f"      - Raw Target: {raw_target if not isinstance(raw_target, list) else str(raw_target[:10]) + '...'}")

        # B. COLLATOR INTERNALS
        # We perform the steps manually using the backbone's tokenizer 
        # to show exactly what is happening inside multitask_collate_fn
        print(f"  [B] Processing Steps (Trace):")
        
        # Step 1: Spacing (Crucial for ProtBert)
        spaced_seq = " ".join(list(raw_seq))
        print(f"      - Step 1 (Spacing): '{spaced_seq[:50]}...'")
        
        # Step 2: Tokenization
        encoded = tokenizer(spaced_seq, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        input_ids = encoded['input_ids'][0]
        
        print(f"      - Step 2 (Tokenization):")
        print(f"        - Tensor Shape: {input_ids.shape}")
        print(f"        - First 5 IDs: {input_ids[:5].tolist()}")
        
        # Step 3: Decoding (Visual Confirmation)
        decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(f"        - Decoded: {decoded_tokens[:5]} ... {decoded_tokens[-1]}")
        
        # Logic Check
        if decoded_tokens[1].startswith("##"):
             print("        ❌ WARNING: Tokens look sub-worded (e.g., ##A). Spacing might be wrong!")
        else:
             print("        ✅ CHECK: Tokens are individual amino acids.")

        # C. FINAL COLLATE FUNCTION OUTPUT
        # Now we run the actual function that Engine uses
        print(f"  [C] Final Collate Output (The Tensor):")
        batch = [raw_sample]
        collated = multitask_collate_fn(batch, tokenizer)
        
        target_tensor = collated['targets']
        print(f"      - Target Tensor Shape: {target_tensor.shape}")
        print(f"      - Target Tensor Type:  {target_tensor.dtype}")
        
        # Specific Task Checks
        if task_name == 'SecondaryStructure':
            # Check alignment (Input length vs Target length)
            # Input includes [CLS] and [SEP], Target usually doesn't map to those
            print(f"      - Alignment Check: Input {input_ids.shape[0]} vs Target {target_tensor.shape[1]}")
            # Target should be padded to match input, with -100 for special tokens
            non_ignored = (target_tensor != -100).sum().item()
            print(f"      - Valid Labels: {non_ignored} (Should match raw seq len {len(raw_seq)})")
            
        elif task_name == 'Thermostability':
            print(f"      - Value: {target_tensor[0].item():.4f}")

    print_section("INSPECTION COMPLETE")

if __name__ == "__main__":
    inspect_pipeline()