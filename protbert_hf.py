import os
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
import torch.nn.functional as F


class ProtBert(nn.Module):
    
    def __init__(self, model_name="Rostlab/prot_bert_bfd", readout="pooler", freeze_bert=False):
        super(ProtBert, self).__init__()
        
        # Load model and tokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.output_dim = self.model.config.hidden_size  
        
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Readout function
        if readout == "pooler":
            self.readout = "pooler"
        elif readout == "sum":
            self.readout = "sum"
        elif readout == "mean":
            self.readout = "mean"
        else:
            raise ValueError("Unknown readout `%s`" % readout)
    
    def forward(self, batch):
        """
        Forward pass for the ProtBert model
        
        Args:
            batch: Dictionary containing 'sequence' and other data
            
        Returns:
            Dictionary with graph_feature and residue_feature
        """
        if isinstance(batch, dict) and 'sequence' in batch:
            sequences = batch['sequence']
        elif hasattr(batch, 'sequence'):
            sequences = batch.sequence
        else:
            # Try to extract from graph if available
            sequences = self.extract_sequence_from_graph(batch)
        
        # Handle single sequence vs batch
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # Tokenize sequences
        # ProtBert expects space-separated amino acids
        spaced_sequences = [' '.join(seq) for seq in sequences]
        
        # Tokenize with padding and truncation
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
            # Mean pooling over sequence length (excluding padding)
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            masked_hidden = last_hidden_state * attention_mask
            graph_feature = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
        elif self.readout == "sum":
            # Sum pooling over sequence length (excluding padding)
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            graph_feature = (last_hidden_state * attention_mask).sum(dim=1)
        else:
            # Use CLS token (first token)
            graph_feature = last_hidden_state[:, 0]
        
        # For residue-level features, remove special tokens
        # [CLS] sequence [SEP] -> sequence
        residue_feature = last_hidden_state[:, 1:-1]  # Remove [CLS] and [SEP]
        
        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature,
            "attention_mask": encoded['attention_mask'][:, 1:-1]  # For residue-level mask
        }
    
    def extract_sequence_from_graph(self, batch):
        """Extract amino acid sequence from graph representation"""
        if hasattr(batch, 'graph') and 'residue_type' in batch['graph']:
            residue_types = batch['graph']['residue_type']
        elif isinstance(batch, dict) and 'graph' in batch:
            residue_types = batch['graph']['residue_type']
        else:
            raise ValueError("Cannot extract sequence from batch")
        
        # Map residue types back to amino acids
        id_to_aa = {
            0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
            10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V',
            20: 'X', 21: 'U', 22: 'B', 23: 'Z', 24: 'O'
        }
        
        if residue_types.dim() == 1:
            # Single sequence
            sequence = ''.join([id_to_aa.get(idx.item(), 'X') for idx in residue_types])
            return sequence
        else:
            # Batch of sequences
            sequences = []
            for seq_residues in residue_types:
                sequence = ''.join([id_to_aa.get(idx.item(), 'X') for idx in seq_residues])
                sequences.append(sequence)
            return sequences


class ProtBertForSequenceClassification(nn.Module):
    """ProtBert model for sequence classification tasks"""
    
    def __init__(self, model_name="Rostlab/prot_bert_bfd", num_labels=1, readout="pooler", 
                 freeze_bert=False, dropout=0.1):
        super().__init__()
        
        self.protbert = ProtBert(model_name, readout, freeze_bert)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.protbert.output_dim, num_labels)
        
    def forward(self, batch):
        outputs = self.protbert(batch)
        graph_feature = outputs["graph_feature"]
        
        graph_feature = self.dropout(graph_feature)
        logits = self.classifier(graph_feature)
        
        return {
            "logits": logits,
            "graph_feature": graph_feature,
            "residue_feature": outputs["residue_feature"]
        }


class ProtBertForTokenClassification(nn.Module):
    """ProtBert model for token (residue) classification tasks"""
    
    def __init__(self, model_name="Rostlab/prot_bert_bfd", num_labels=8, readout="mean", 
                 freeze_bert=False, dropout=0.1):
        super().__init__()
        
        self.protbert = ProtBert(model_name, readout, freeze_bert)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.protbert.output_dim, num_labels)
        
    def forward(self, batch):
        outputs = self.protbert(batch)
        residue_feature = outputs["residue_feature"]
        
        # Apply dropout and classification
        residue_feature = self.dropout(residue_feature)
        logits = self.classifier(residue_feature)
        
        return {
            "logits": logits,
            "graph_feature": outputs["graph_feature"],
            "residue_feature": residue_feature,
            "attention_mask": outputs["attention_mask"]
        }


# LoRA adapter implementation
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
        # Original transformation
        original_out = self.linear(x)
        
        # LoRA transformation
        lora_out = self.lora_B(self.dropout(self.lora_A(x)))
        
        # Combine with scaling
        return original_out + (self.alpha / self.rank) * lora_out


class ProtBertWithLoRA(nn.Module):
    """ProtBert model with LoRA adapters"""
    
    def __init__(self, model_name="Rostlab/prot_bert_bfd", num_labels=1, readout="pooler",
                 lora_rank=16, lora_alpha=32, lora_dropout=0.1, task_type="classification"):
        super().__init__()
        
        self.protbert = ProtBert(model_name, readout, freeze_bert=True)
        self.task_type = task_type
        
        # Add LoRA adapters to attention layers
        self.add_lora_adapters(lora_rank, lora_alpha, lora_dropout)
        
        # Task-specific head
        if task_type == "classification":
            self.head = nn.Sequential(
                nn.Dropout(lora_dropout),
                nn.Linear(self.protbert.output_dim, num_labels)
            )
        elif task_type == "token_classification":
            self.head = nn.Sequential(
                nn.Dropout(lora_dropout),
                nn.Linear(self.protbert.output_dim, num_labels)
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def add_lora_adapters(self, rank, alpha, dropout):
        """Add LoRA adapters to BERT attention layers"""
        for layer in self.protbert.model.encoder.layer:
            # Replace query, key, value projections with LoRA versions
            attention = layer.attention.self
            
            # Store original weights and replace with LoRA
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
                    # Copy original weights
                    lora_layer.linear.weight.data = original_layer.weight.data.clone()
                    setattr(attention, name, lora_layer)
    
    def forward(self, batch):
        outputs = self.protbert(batch)
        
        if self.task_type == "classification":
            graph_feature = outputs["graph_feature"]
            logits = self.head(graph_feature)
        elif self.task_type == "token_classification":
            residue_feature = outputs["residue_feature"]
            logits = self.head(residue_feature)
        
        return {
            "logits": logits,
            "graph_feature": outputs["graph_feature"],
            "residue_feature": outputs["residue_feature"]
        }

def create_protbert_model(model_type="base", **kwargs):
    """Factory function to create ProtBert models"""
    models = {
        "base": ProtBert,
        "classification": ProtBertForSequenceClassification,
        "token_classification": ProtBertForTokenClassification,
        "lora": ProtBertWithLoRA
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)