import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

class SharedProtBert(nn.Module):
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
        else:
            print("[SharedProtBert] Backbone is completely frozen (Pure LoRA).")

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

    def get_cls_embedding(self, device):
        """
        Returns the raw embedding of the [CLS] token from the base model.
        Used for Semantic Initialization of Prompts.
        """
        cls_id = self.tokenizer.cls_token_id
        token_tensor = torch.tensor([cls_id], device=device)
        
        # We look up the embedding from the base HuggingFace model (before LoRA)
        with torch.no_grad():
            embedding = self.base_model.embeddings.word_embeddings(token_tensor)
        
        # Returns shape [1, 1024]
        return embedding

    def forward(self, input_ids, attention_mask, task_prompt_embeds=None, task_type='token', debug=False):
        # A. GET RAW EMBEDDINGS
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # B. PREPEND PROMPT (Early Fusion)
        if task_prompt_embeds is not None:
            inputs_embeds = torch.cat([task_prompt_embeds, inputs_embeds], dim=1)
            
            batch_size = inputs_embeds.shape[0]
            prompt_mask = torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
            
            if debug:
                print(f"    [Backbone] Prompt injected. New Seq Shape: {inputs_embeds.shape}")

        # C. FORWARD PASS
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # D. SLICING & OUTPUT HANDLING
        if task_prompt_embeds is not None:
            sequence_output = sequence_output[:, 1:, :]
            attention_mask = attention_mask[:, 1:]

        if task_type == 'token':
            return sequence_output
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            if debug:
                print(f"    [Backbone] Mean Pooling applied on sliced sequence. Shape: {pooled.shape}")
                
            return pooled