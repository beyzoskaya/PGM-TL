import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

class SharedProtBert(nn.Module):

    def __init__(self, model_name="Rostlab/prot_bert_bfd", readout="mean",
                 lora=True, lora_rank=32, lora_alpha=32, lora_dropout=0.1,
                 freeze_backbone=True, unfrozen_layers=4, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.readout = readout

        # Load base ProtBERT
        self.model = AutoModel.from_pretrained(model_name)
        self.pool_norm = nn.LayerNorm(self.model.config.hidden_size)

        # Freeze backbone if requested
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
            # Unfreeze top N layers
            self.unfreeze_top_layers(unfrozen_layers)

        # Apply LoRA via PEFT
        if lora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["query", "key", "value", "dense"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.model = get_peft_model(self.model, lora_config)

        # Print summary
        if self.verbose:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[SharedProtBert] model_name={model_name}")
            print(f"[SharedProtBert] total params={total_params:,}, trainable params={trainable_params:,}")

    def unfreeze_top_layers(self, num_layers_to_unfreeze=4):
 
        total_layers = len(self.model.encoder.layer)
        for i, layer in enumerate(self.model.encoder.layer):
            requires_grad = i >= total_layers - num_layers_to_unfreeze
            for p in layer.parameters():
                p.requires_grad = requires_grad

        # Optional: also unfreeze pooler if exists
        if hasattr(self.model, "pooler") and self.model.pooler is not None:
            for p in self.model.pooler.parameters():
                p.requires_grad = True
        if self.verbose:
            print(f"[SharedProtBert] Unfroze top {num_layers_to_unfreeze} layers of backbone.")

    def forward(self, input_ids, attention_mask=None, per_residue=False):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        if per_residue:
            return last_hidden

        # Sequence-level pooling
        if self.readout == "mean":
            pooled = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        else:  # cls token
            pooled = last_hidden[:, 0, :]

        return self.pool_norm(pooled)

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

def build_regression_head(hidden_dim, num_labels=1, dropout_rate=0.2):
    return nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim // 2, num_labels)
    )

def build_token_classification_head(hidden_dim, num_labels, dropout_rate=0.3):
    return nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, num_labels)
    )

def build_sequence_classification_head(hidden_dim, num_labels, dropout_rate=0.2):
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, num_labels)
    )
