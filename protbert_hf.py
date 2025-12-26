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

class SharedProtBertPromptTuning(nn.Module):
    def __init__(self, model_name="Rostlab/prot_bert_bfd", 
                 lora_rank=16, 
                 lora_alpha=32, 
                 lora_dropout=0.1,
                 unfrozen_layers=2):
        super().__init__()
        
        print(f"[SharedProtBert] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.base_model.gradient_checkpointing_enable()

        # 1. FREEZE EVERYTHING
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 2. SELECTIVE UNFREEZING
        if unfrozen_layers > 0:
            print(f"[SharedProtBert] Unfreezing top {unfrozen_layers} transformer layers...")
            layers_to_unfreeze = self.base_model.encoder.layer[-unfrozen_layers:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            
            if hasattr(self.base_model, 'pooler') and self.base_model.pooler is not None:
                for param in self.base_model.pooler.parameters():
                    param.requires_grad = True

        # 3. APPLY LORA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=lora_rank, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "dense"]
        )
        self.model = get_peft_model(self.base_model, peft_config)
        self.hidden_size = self.base_model.config.hidden_size
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"[SharedProtBert] Trainable Parameters: {trainable_params:,} || ({100 * trainable_params / all_params:.2f}%)")

    def forward(self, input_ids, attention_mask, task_prompt_embeds=None, task_type='token', debug=False):
        """
        Arguments:
            input_ids: [Batch, Seq_Len]
            attention_mask: [Batch, Seq_Len]
            task_prompt_embeds: [Batch, 1, Hidden] - The learnable task vector
            task_type: 'token' (returns sequence) or 'sequence' (returns pooled embedding)
        """
        
        # A. GET RAW EMBEDDINGS
        # We access the base model to convert IDs -> Vectors
        # self.model is the PeftModel, self.model.base_model is the underlying wrapper
        # We need the original Hugging Face model's embeddings.
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # B. PREPEND PROMPT (Early Fusion)
        if task_prompt_embeds is not None:
            # 1. Concatenate Prompt + Sequence
            # Shape: [Batch, 1 + Seq_Len, Hidden]
            inputs_embeds = torch.cat([task_prompt_embeds, inputs_embeds], dim=1)
            
            # 2. Extend Attention Mask
            # We add a '1' to the start of the mask so the model attends to the prompt
            batch_size = inputs_embeds.shape[0]
            prompt_mask = torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
            
            if debug:
                print(f"    [Backbone] Prompt injected. New Seq Shape: {inputs_embeds.shape}")

        # C. FORWARD PASS
        # We pass 'inputs_embeds' instead of 'input_ids'
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # D. SLICING & OUTPUT HANDLING
        if task_prompt_embeds is not None:
            # CRITICAL: We remove the prompt token from the output.
            # The prompt has done its job (contextualizing the sequence).
            # We don't want to classify the prompt, nor include it in Mean Pooling.
            sequence_output = sequence_output[:, 1:, :]
            # We also revert the mask for pooling purposes
            attention_mask = attention_mask[:, 1:]

        if task_type == 'token':
            # Return full sequence (Batch, Seq_Len, Hidden)
            # Perfect for SSP ( Secondary Structure )
            return sequence_output
            
        else:
            # Mean Pooling for Sequence Tasks (Thermo, Cloning)
            # We use the original sequence mask (excluding prompt)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            if debug:
                print(f"    [Backbone] Mean Pooling applied on sliced sequence. Shape: {pooled.shape}")
                
            return pooled
