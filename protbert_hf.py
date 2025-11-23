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
        self.base_model.gradient_checkpointing_enable()

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

    def forward(self, input_ids, attention_mask, task_type, debug=False):
        """
        task_type: 
            - 'token': Returns (Batch, Seq_Len, Hidden) for Secondary Structure
            - 'sequence': Returns (Batch, Hidden) for Regression/Classification
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        if debug:
            print(f"    [Backbone Debug] Input IDs Shape: {input_ids.shape}")
            print(f"    [Backbone Debug] Raw Embeddings: Mean={last_hidden_state.mean().item():.4f} | Std={last_hidden_state.std().item():.4f}")

        if task_type == 'token': 
            return last_hidden_state 
        
        else: 
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            if debug:
                print(f"    [Backbone Debug] Pooled Embedding (Sequence): Mean={pooled.mean().item():.4f} | Std={pooled.std().item():.4f}")
                
            return pooled

