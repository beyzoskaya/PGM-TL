import os
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class ProtBert(nn.Module):
    
    def __init__(self, model_name="Rostlab/prot_bert_bfd", readout="pooler", freeze_bert=False):
        super(ProtBert, self).__init__()
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.output_dim = self.model.config.hidden_size  
        
        if freeze_bert:
            # Freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze the last 2 transformer encoder layers
            for layer in self.model.encoder.layer[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
            if hasattr(self.model, "pooler"):
                for param in self.model.pooler.parameters():
                    param.requires_grad = True

        if readout in ["pooler", "sum", "mean"]:
            self.readout = readout
        else:
            raise ValueError(f"Unknown readout `{readout}`")
    
    def forward(self, batch):
        if isinstance(batch, dict) and 'sequence' in batch:
            sequences = batch['sequence']
        elif hasattr(batch, 'sequence'):
            sequences = batch.sequence
        else:
            sequences = self.extract_sequence_from_graph(batch)
 
        if isinstance(sequences, str):
            sequences = [sequences]
        
        spaced_sequences = [' '.join(seq) for seq in sequences]
        
        encoded = self.tokenizer(
            spaced_sequences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024, 
            return_tensors='pt'
        )
        
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Forward pass through BERT
        outputs = self.model(**encoded)
        
        # Extract features
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Get graph-level features
        if self.readout == "pooler" and hasattr(outputs, 'pooler_output'):
            graph_feature = outputs.pooler_output
        elif self.readout == "mean":
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            masked_hidden = last_hidden_state * attention_mask
            graph_feature = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
        elif self.readout == "sum":
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            graph_feature = (last_hidden_state * attention_mask).sum(dim=1)
        else:
            graph_feature = last_hidden_state[:, 0]
        
        # For residue-level features, remove special tokens [CLS] and [SEP]
        residue_feature = last_hidden_state[:, 1:-1]
        
        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature,
            "attention_mask": encoded['attention_mask'][:, 1:-1]
        }
    
    def extract_sequence_from_graph(self, batch):
        """Extract amino acid sequence from graph representation"""
        if hasattr(batch, 'graph') and 'residue_type' in batch['graph']:
            residue_types = batch['graph']['residue_type']
        elif isinstance(batch, dict) and 'graph' in batch:
            residue_types = batch['graph']['residue_type']
        else:
            raise ValueError("Cannot extract sequence from batch")
        
        id_to_aa = {
            0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
            10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V',
            20: 'X', 21: 'U', 22: 'B', 23: 'Z', 24: 'O'
        }
        
        if residue_types.dim() == 1:
            sequence = ''.join([id_to_aa.get(idx.item(), 'X') for idx in residue_types])
            return sequence
        else:
            sequences = []
            for seq_residues in residue_types:
                sequence = ''.join([id_to_aa.get(idx.item(), 'X') for idx in seq_residues])
                sequences.append(sequence)
            return sequences

class LoRALinear(nn.Module):
    """LoRA (Low-Rank Adaptation) layer"""
    
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        original_out = self.linear(x)
        lora_out = self.lora_B(self.dropout(self.lora_A(x)))
        return original_out + (self.alpha / self.rank) * lora_out


class ProtBertWithLoRA(nn.Module):
    """ProtBert with LoRA adapters for efficient fine-tuning"""

    def __init__(self, model_name="Rostlab/prot_bert_bfd", readout="mean",
                 lora_rank=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        
        self.protbert = ProtBert(model_name, readout, freeze_bert=True)

        trainable = [n for n, p in self.protbert.model.named_parameters() if p.requires_grad]
        print(f"Trainable ProtBert parameters ({len(trainable)}):")
        for n in trainable:
            print("  ", n)
            
        self.output_dim = self.protbert.output_dim  # Expose output_dim
        
        # Add LoRA adapters to attention layers
        self.add_lora_adapters(lora_rank, lora_alpha, lora_dropout)
    
    def add_lora_adapters(self, rank, alpha, dropout):
        """Add LoRA adapters to all attention layers"""
        for layer in self.protbert.model.encoder.layer:
            attention = layer.attention.self
            
            for name in ['query', 'key', 'value']:
                if hasattr(attention, name):
                    original_layer = getattr(attention, name)
                    lora_layer = LoRALinear(
                        original_layer.in_features,
                        original_layer.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout
                    )
                    # Copy original weights to LoRA's frozen layer
                    lora_layer.linear.weight.data = original_layer.weight.data.clone()
                    setattr(attention, name, lora_layer)
    
    def forward(self, batch):
        """Forward pass through ProtBert backbone"""
        outputs = self.protbert(batch)
        return outputs