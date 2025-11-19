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

        # Base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # LoRA matrices (A,B)
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.A)
        nn.init.zeros_(self.B)

    def forward(self, x):
        lora_update = (self.dropout(x) @ self.A.T @ self.B.T) * self.scaling
        return x @ self.weight.T + lora_update + (self.bias if self.bias is not None else 0)


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
# ProtBert With LoRA (DEBUG FRIENDLY)
# ============================================================
class ProtBertWithLoRA(nn.Module):
    def __init__(self, model_name="Rostlab/prot_bert_bfd", readout="mean",
                 lora_rank=16, lora_alpha=32, lora_dropout=0.1, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.bert = AutoModel.from_pretrained(model_name)
        self.readout = readout

        # Freeze backbone
        for p in self.bert.parameters():
            p.requires_grad = False

        # Inject LoRA layers
        self.inject_lora_correct(lora_rank, lora_alpha, lora_dropout)

        if self.verbose:
            # Summary of trainable parameters
            total_trainable = 0
            print("\n[LoRA DEBUG] Trainable parameters:")
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"  ✓ {name} {p.shape}")
                    total_trainable += p.numel()
            print(f"[LoRA DEBUG] Total trainable: {total_trainable}\n")

    def inject_lora_correct(self, r, alpha, dropout):
        if self.verbose:
            print("\n[LoRA DEBUG] Injecting LoRA into ProtBert layers...")
        for layer_idx, layer in enumerate(self.bert.encoder.layer):
            for attr in ['query', 'key', 'value']:
                old = getattr(layer.attention.self, attr)
                new = LoRALinear(old.in_features, old.out_features,
                                 r=r, alpha=alpha, dropout=dropout,
                                 bias=(old.bias is not None))
                with torch.no_grad():
                    new.weight.copy_(old.weight)
                    if old.bias is not None:
                        new.bias.copy_(old.bias)
                setattr(layer.attention.self, attr, new)
                if self.verbose:
                    print(f"  Injected LoRA: encoder.layer[{layer_idx}].attention.self.{attr}")

            # output.dense
            old = layer.attention.output.dense
            new = LoRALinear(old.in_features, old.out_features,
                             r=r, alpha=alpha, dropout=dropout,
                             bias=(old.bias is not None))
            with torch.no_grad():
                new.weight.copy_(old.weight)
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            layer.attention.output.dense = new
            if self.verbose:
                print(f"  Injected LoRA: encoder.layer[{layer_idx}].attention.output.dense")
        if self.verbose:
            print("[LoRA DEBUG] Injection complete.\n")

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        if self.readout == "mean":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        return last_hidden[:, 0, :]


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
