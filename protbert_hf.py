import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import math
import warnings

try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False
    warnings.warn("peft not available. The code will still work but LoRA via PEFT won't be used.")

class SharedProtBert(nn.Module):
    """
    Shared ProtBERT backbone for multitask training.

    - If use_lora=True and peft is installed, we attach LoRA via PEFT.
    - forward(..., return_per_token=True) -> returns (B, L, H) last_hidden_state
    - forward(..., return_per_token=False) -> returns (B, H) pooled (mean) embedding
    - Provides helper methods to inspect trainable params.
    """

    def __init__(self,
                 model_name="Rostlab/prot_bert_bfd",
                 readout="mean",
                 use_lora=True,
                 lora_r=16,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 freeze_backbone=True,
                 lora_target_modules=None,
                 device=None,
                 verbose=False):
        super().__init__()
        self.verbose = verbose
        self.model_name = model_name
        self.readout = readout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, add_prefix_space=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

        # Default LoRA target modules for BERT-like models
        if lora_target_modules is None:
            # These module name fragments work for many BERT-like implementations:
            lora_target_modules = ["query", "key", "value", "dense", "output_dense"]

        self.use_lora = use_lora and _HAS_PEFT
        self._lora_attached = False

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        if self.use_lora:
            # Attach LoRA via PEFT
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"  # not used strongly here, but required param historically — benign
            )
            try:
                self.model = get_peft_model(self.model, lora_config)
                self._lora_attached = True
                if self.verbose:
                    print("[SharedProtBert] LoRA attached via PEFT. Target modules:", lora_target_modules)
            except Exception as e:
                warnings.warn(f"PEFT get_peft_model failed: {e}. Proceeding without PEFT LoRA.")
                self._lora_attached = False
                self.use_lora = False

        # small layernorm to stabilize pooled embeddings (optional)
        self.pool_norm = nn.LayerNorm(self.model.config.hidden_size)

        if self.verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[SharedProtBert] model_name={model_name}")
            print(f"[SharedProtBert] total params={total_params:,}, trainable params={trainable_params:,}")

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    def forward(self, input_ids, attention_mask=None, return_per_token=False):
        """
        Return either token-level embeddings (B, L, H) if return_per_token=True
        or pooled (B, H) otherwise.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        if return_per_token:
            return last_hidden

        # pooled mean over non-masked tokens
        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            # ensure float mask
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / denom

        pooled = self.pool_norm(pooled)
        return pooled

    def get_trainable_param_names(self, max_display=50):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return names[:max_display]

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def unfreeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def to(self, device):
        self.device = device
        super().to(device)
        try:
            self.model.to(device)
        except Exception:
            pass
        return self
