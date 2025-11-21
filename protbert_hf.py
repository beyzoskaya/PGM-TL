import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

class SharedProtBert(nn.Module):
    def __init__(self, model_name="Rostlab/prot_bert_bfd", 
                 lora_rank=16, 
                 lora_alpha=32, 
                 lora_dropout=0.1,
                 unfrozen_layers=2): # Number of top layers to unfreeze
        super().__init__()
        
        print(f"[SharedProtBert] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.base_model = AutoModel.from_pretrained(model_name)

        # 1. FREEZING LOGIC
        # First, freeze everything
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 2. SELECTIVE UNFREEZING
        if unfrozen_layers > 0:
            print(f"[SharedProtBert] Unfreezing top {unfrozen_layers} transformer layers...")
            # The encoder layers are usually stored in base_model.encoder.layer
            # We slice the last N layers
            layers_to_unfreeze = self.base_model.encoder.layer[-unfrozen_layers:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Also unfreeze the Pooler layer (crucial for sequence classification)
            if hasattr(self.base_model, 'pooler') and self.base_model.pooler is not None:
                for param in self.base_model.pooler.parameters():
                    param.requires_grad = True

        # 3. APPLY LORA
        # This adds adapter layers to the frozen (and unfrozen) parts
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=lora_rank, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "dense"]
        )
        self.model = get_peft_model(self.base_model, peft_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"[SharedProtBert] Trainable Parameters: {trainable_params:,} || ({100 * trainable_params / all_params:.2f}%)")

    @property
    def hidden_size(self):
        return self.base_model.config.hidden_size

    def forward(self, input_ids, attention_mask, task_type):
        """
        task_type: 
            - 'token': Returns (Batch, Seq_Len, Hidden) for Secondary Structure
            - 'sequence': Returns (Batch, Hidden) for Regression/Classification
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        if task_type == 'token':
            # Return per-residue embeddings
            return last_hidden_state
        
        else:
            # Mean Pooling (Attention-Weighted)
            # This is mathematically better than just taking the [CLS] token for proteins
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            return pooled

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
