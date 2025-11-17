# protbert_hf.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
import copy
import math

class LoRALinear(nn.Module):
    """
    LoRA applied to linear layers: W = W_0 + A @ B
    """
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # LoRA matrices
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)

        # Init weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.A)
        nn.init.zeros_(self.B)

    def forward(self, x):
        lora_update = self.dropout(x) @ self.A.T @ self.B.T * self.scaling
        return x @ self.weight.T + lora_update + (self.bias if self.bias is not None else 0)

class ProtBert(nn.Module):
    def __init__(self, model_name='Rostlab/prot_bert_bfd', readout='mean', freeze_bert=True):
        """
        Base ProtBert model wrapper.
        readout: 'mean' or 'cls'
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.bert = AutoModel.from_pretrained(model_name)
        self.readout = readout

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, D]
        if self.readout == 'mean':
            seq_emb = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:  # 'cls'
            seq_emb = last_hidden[:, 0, :]
        return seq_emb  # [B, D]

class ProtBertWithLoRA(nn.Module):
    def __init__(self, model_name='Rostlab/prot_bert_bfd', readout='mean', lora_rank=16, lora_alpha=32, lora_dropout=0.1):
        """
        ProtBert with LoRA adapters.
        Freezes backbone and adds LoRA to attention projections.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.bert = AutoModel.from_pretrained(model_name)
        self.readout = readout

        # Freeze backbone
        for param in self.bert.parameters():
            param.requires_grad = False

        # Inject LoRA into attention layers
        self.inject_lora(lora_rank, lora_alpha, lora_dropout)

    def inject_lora(self, r, alpha, dropout):
        """
        Replace Bert attention linear layers with LoRA-augmented versions.
        Only inject into query/key/value layers for efficiency.
        """
        for name, module in self.bert.named_modules():
            if isinstance(module, nn.Linear) and 'attention' in name:
                # Wrap linear layer with LoRA
                lora_linear = LoRALinear(module.in_features, module.out_features, r=r, alpha=alpha, dropout=dropout, bias=(module.bias is not None))
                # Copy original weights
                with torch.no_grad():
                    lora_linear.weight.copy_(module.weight)
                    if module.bias is not None:
                        lora_linear.bias.copy_(module.bias)
                # Replace in model
                parent = self.get_parent_module(name)
                attr_name = name.split('.')[-1]
                setattr(parent, attr_name, lora_linear)

    def get_parent_module(self, name):
        """
        Retrieve the parent module to replace attribute.
        """
        parts = name.split('.')
        module = self.bert
        for p in parts[:-1]:
            module = getattr(module, p)
        return module

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        if self.readout == 'mean':
            seq_emb = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            seq_emb = last_hidden[:, 0, :]
        return seq_emb

class SharedProtBert(nn.Module):
    """
    Shared ProtBert backbone for multi-task learning.
    Optional LoRA.
    """
    def __init__(self, model_name='Rostlab/prot_bert_bfd', readout='mean',
                 lora=False, lora_rank=16, lora_alpha=32, lora_dropout=0.1,
                 freeze_backbone=True):
        super().__init__()

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)

        # --- Backbone ---
        if lora:
            self.backbone = ProtBertWithLoRA(model_name=model_name, readout=readout,
                                             lora_rank=lora_rank, lora_alpha=lora_alpha,
                                             lora_dropout=lora_dropout)
        else:
            self.backbone = ProtBert(model_name=model_name, readout=readout,
                                     freeze_bert=freeze_backbone)

    def forward(self, input_ids, attention_mask, per_residue=False):
        """
        per_residue: if True, return embeddings for each token [B, L, D]
                    if False, return pooled embeddings [B, D]
        """
        outputs = self.backbone.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, D]
        if per_residue:
            return last_hidden
        if self.backbone.readout == 'mean':
            seq_emb = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:  # 'cls'
            seq_emb = last_hidden[:, 0, :]
        return seq_emb

    @property
    def hidden_size(self):
        return self.backbone.bert.config.hidden_size

#if __name__ == "__main__":
#    import torch

#    sample_seq = ["MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAY"]
#    model = SharedProtBert(lora=True)
#    tokenizer = model.backbone.tokenizer
#    enc = tokenizer(sample_seq, return_tensors="pt", padding=True)
#    emb = model(enc['input_ids'], enc['attention_mask'])
#    print("SharedProtBert output shape:", emb.shape)
#    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
