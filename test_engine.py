from flip_hf import Thermostability
from protbert_hf import SharedProtBert
from engine_hf_with_task_specific_encoder import MultiTaskEngine
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a small subset of dataset
dataset = Thermostability(path='./data')
subset = [dataset[i] for i in range(5)]  # first 5 sequences

# Load backbone
backbone = SharedProtBert().to(device)

# Initialize engine with one dataset
engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=[{'type':'regression', 'num_labels':1}],
    train_sets=[subset],
    valid_sets=[subset],
    test_sets=[subset],
    batch_size=2,
    device=device
)

# Test collate + print
batch_encoding, batch_targets = engine.print_first_batch(split='train', dataset_idx=0)
from flip_hf import Thermostability
from protbert_hf import SharedProtBert
from engine_hf_with_task_specific_encoder import MultiTaskEngine
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a small subset of dataset
dataset = Thermostability(path='./data')
subset = [dataset[i] for i in range(5)]  # first 5 sequences

# Load backbone
backbone = SharedProtBert().to(device)

# Initialize engine with one dataset
engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=[{'type':'regression', 'num_labels':1}],
    train_sets=[subset],
    valid_sets=[subset],
    test_sets=[subset],
    batch_size=2,
    device=device
)

# Test collate + print
batch_encoding, batch_targets = engine.print_first_batch(split='train', dataset_idx=0)
