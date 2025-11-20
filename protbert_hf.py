import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import math

# ============================================================
# LoRA Layer 
# ============================================================
class LoRALinear(nn.Module):
    """LoRA applied to linear layers: W = W_0 + (dropout(x) @ A^T @ B^T) * scaling"""
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Base weight (FROZEN)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False  # CRITICAL: Freeze base weights

        # Optional bias (FROZEN)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.bias.requires_grad = False  # CRITICAL: Freeze bias
        else:
            self.register_parameter("bias", None)

        # LoRA matrices (A, B) - TRAINABLE
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)

        # ✓ FIXED: Initialize A with small random values (not zeros!)
        nn.init.normal_(self.A, std=0.02)
        # Keep B at zero (standard LoRA practice)
        nn.init.zeros_(self.B)

    def forward(self, x):
        # lora_update = x @ A.T @ B.T * scaling
        lora_update = (self.dropout(x) @ self.A.T @ self.B.T) * self.scaling
        
        # Output = base_weight + lora_update + bias
        output = x @ self.weight.T + lora_update
        if self.bias is not None:
            output = output + self.bias
        
        return output


# ============================================================
# ProtBert (baseline)
# ============================================================
class ProtBert(nn.Module):
    def __init__(self, model_name="Rostlab/prot_bert_bfd", readout="mean", freeze_bert=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.bert = AutoModel.from_pretrained(model_name)
        self.readout = readout

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        if self.readout == "mean":
            seq_emb = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        else:
            seq_emb = last_hidden[:, 0, :]
        return seq_emb


# ============================================================
# ProtBert With LoRA - FIXED (Gradient Flow)
# ============================================================
class ProtBertWithLoRA(nn.Module):
    def __init__(self, model_name="Rostlab/prot_bert_bfd", readout="mean",
                 lora_rank=16, lora_alpha=32, lora_dropout=0.1, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.bert = AutoModel.from_pretrained(model_name)
        self.readout = readout

        # Freeze BERT backbone
        for p in self.bert.parameters():
            p.requires_grad = False

        # Inject LoRA layers
        self.inject_lora_correct(lora_rank, lora_alpha, lora_dropout)

        if self.verbose:
            # Summary of trainable parameters
            total_trainable = 0
            lora_params = 0
            print("\n[LoRA DEBUG] Trainable parameters after injection:")
            for name, p in self.named_parameters():
                if p.requires_grad:
                    param_count = p.numel()
                    total_trainable += param_count
                    if 'A' in name or 'B' in name:
                        lora_params += param_count
                    if 'layer.0' in name:  # Only show layer 0 as example
                        print(f"  ✓ {name} {p.shape}")
            print(f"[LoRA DEBUG] Total trainable: {total_trainable:,} (LoRA only)")
            print(f"[LoRA DEBUG] LoRA parameters: {lora_params:,}\n")

    def inject_lora_correct(self, r, alpha, dropout):
        if self.verbose:
            print("\n[LoRA DEBUG] Injecting LoRA into ProtBert attention layers...")
        
        lora_count = 0
        for layer_idx, layer in enumerate(self.bert.encoder.layer):
            # Attention heads: query, key, value
            for attr in ['query', 'key', 'value']:
                old = getattr(layer.attention.self, attr)
                new = LoRALinear(old.in_features, old.out_features,
                                 r=r, alpha=alpha, dropout=dropout,
                                 bias=(old.bias is not None))
                with torch.no_grad():
                    new.weight.copy_(old.weight)
                    if old.bias is not None and new.bias is not None:
                        new.bias.copy_(old.bias)
                
                setattr(layer.attention.self, attr, new)
                lora_count += 1

            # Attention output dense layer
            old = layer.attention.output.dense
            new = LoRALinear(old.in_features, old.out_features,
                             r=r, alpha=alpha, dropout=dropout,
                             bias=(old.bias is not None))
            with torch.no_grad():
                new.weight.copy_(old.weight)
                if old.bias is not None and new.bias is not None:
                    new.bias.copy_(old.bias)
            
            layer.attention.output.dense = new
            lora_count += 1
            
            if self.verbose and layer_idx == 0:
                print(f"  Injected LoRA into layer 0 (4 total per layer)")

        if self.verbose:
            print(f"[LoRA DEBUG] Total LoRA modules injected: {lora_count}\n")

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        if self.readout == "mean":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        return last_hidden[:, 0, :]


# ============================================================
# Task-Specific Head Builders
# ============================================================
def build_regression_head(hidden_dim, num_labels=1, dropout_rate=0.2):
    """
    Deeper head for regression tasks (e.g., Thermostability)
    Uses intermediate ReLU layer for better capacity
    """
    return nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim // 2, num_labels)
    )


def build_token_classification_head(hidden_dim, num_labels, dropout_rate=0.3):
    """
    Head for per-residue token classification (e.g., Secondary Structure)
    Applies projection directly to maintain spatial structure
    """
    return nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, num_labels)
    )


def build_sequence_classification_head(hidden_dim, num_labels, dropout_rate=0.2):
    """
    Head for sequence-level classification (e.g., Cloning CLF)
    Simple but effective for binary/multi-class classification
    """
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, num_labels)
    )


# ============================================================
# Shared Backbone 
# ============================================================
class SharedProtBert(nn.Module):
    def __init__(self, model_name="Rostlab/prot_bert_bfd", readout="mean",
                 lora=True, lora_rank=16, lora_alpha=32, lora_dropout=0.1,
                 freeze_backbone=True, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)

        if lora:
            self.backbone = ProtBertWithLoRA(
                model_name=model_name,
                readout=readout,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                verbose=verbose
            )
        else:
            self.backbone = ProtBert(
                model_name=model_name,
                readout=readout,
                freeze_bert=freeze_backbone
            )

    def forward(self, input_ids, attention_mask, per_residue=False):
        if per_residue:
            outputs = self.backbone.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state
        return self.backbone(input_ids, attention_mask)

    @property
    def hidden_size(self):
        return self.backbone.bert.config.hidden_size