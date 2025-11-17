import torch
from flip_hf import Thermostability  # or any dataset you want to test
from protbert_hf import SharedProtBert  # your updated ProtBert model
from engine_hf_with_task_specific_encoder import MultiTaskEngine  # the skeleton

# Load dataset
train_thermo = Thermostability(path='./data').split()[0]  # get train split
valid_thermo = Thermostability(path='./data').split()[1]  # valid split
test_thermo = Thermostability(path='./data').split()[2]  # test split

# Load backbone
backbone = SharedProtBert('Rostlab/prot_bert_bfd')
backbone.to('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize engine
engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=[{'type':'regression','num_labels':1}],
    train_sets=[train_thermo],
    valid_sets=[valid_thermo],
    test_sets=[test_thermo],
    batch_size=2,
    num_worker=0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Test: print first batch
batch = engine.print_first_batch(split='train', dataset_idx=0)
import torch
from flip_hf import Thermostability  # or any dataset you want to test
from protbert_hf import SharedProtBert  # your updated ProtBert model
from engine_hf_with_task_specific_encoder import MultiTaskEngine  # the skeleton

# Load dataset
train_thermo = Thermostability(path='./data').split()[0]  # get train split
valid_thermo = Thermostability(path='./data').split()[1]  # valid split
test_thermo = Thermostability(path='./data').split()[2]  # test split

# Load backbone
backbone = SharedProtBert('Rostlab/prot_bert_bfd')
backbone.to('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize engine
engine = MultiTaskEngine(
    backbone=backbone,
    task_configs=[{'type':'regression','num_labels':1}],
    train_sets=[train_thermo],
    valid_sets=[valid_thermo],
    test_sets=[test_thermo],
    batch_size=2,
    num_worker=0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Test: print first batch
batch = engine.print_first_batch(split='train', dataset_idx=0)
