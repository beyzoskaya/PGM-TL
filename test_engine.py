# test_engine_step2.py
import torch
from flip_hf import Thermostability, SecondaryStructure, CloningCLF
from protbert_hf import SharedProtBert
from engine_hf_with_task_specific_encoder import MultiTaskEngine
from transformers import AutoTokenizer

# Load datasets
train_thermo = Thermostability(path='./data')
train_ss = SecondaryStructure(path='./data')
train_cloning = CloningCLF(path='./data')

train_sets = [train_thermo, train_ss, train_cloning]
valid_sets = [train_thermo, train_ss, train_cloning]
test_sets = [train_thermo, train_ss, train_cloning]

# Load backbone
backbone = SharedProtBert(lora=False)

# Attach tokenizer manually
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
backbone.tokenizer = tokenizer

# Task configs
task_configs = [
    {'type': 'regression', 'num_labels': 1},
    {'type': 'token_classification', 'num_labels': 8},
    {'type': 'binary_classification', 'num_labels': 1}
]

# Initialize engine
engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=task_configs,
    train_sets=train_sets,
    valid_sets=valid_sets,
    test_sets=test_sets,
    batch_size=2,
    device='cpu'
)

# Print first batch from each dataset to verify collate + tokenization
for idx, _ in enumerate(train_sets):
    print(f"\n--- Dataset {idx} first batch ---")
    encoding, targets = engine.print_first_batch(split='train', dataset_idx=idx)
