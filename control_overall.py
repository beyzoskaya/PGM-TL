import torch
from protbert_hf import SharedProtBert

device = "cuda" if torch.cuda.is_available() else "cpu"
bp = SharedProtBert(use_lora=False, freeze_backbone=True, verbose=True, device=device)

print("Hidden size:", bp.hidden_size)
print("Trainable param count:", bp.count_trainable_parameters())
print("Some trainable names:", bp.get_trainable_param_names())

# create fake batch
batch_size = 2
seqs = ["MKTWVTFISLLFLFSS", "ACDEFGHIKLMNPQRSTVWY"]
tok = bp.tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=64)
input_ids = tok["input_ids"].to(device)
attention_mask = tok["attention_mask"].to(device)

# forward token and pooled
tokens = bp(input_ids, attention_mask, return_per_token=True)
pooled = bp(input_ids, attention_mask, return_per_token=False)
print("tokens shape:", tokens.shape)   # (B, L, H)
print("pooled shape:", pooled.shape)   # (B, H)

# quick backward check through heads (dummy head)
head = torch.nn.Linear(bp.hidden_size, 1).to(device)
opt = torch.optim.Adam(list(bp.model.parameters()) + list(head.parameters()), lr=1e-4)

head.train()
opt.zero_grad()
out = head(pooled)  # shape (B,1)
loss = ((out - torch.randn_like(out))**2).mean()
loss.backward()
print("After backward, some grad norms (first 10 trainable params):")
cnt=0
for n,p in bp.model.named_parameters():
    if p.requires_grad:
        if p.grad is None:
            print(n, "grad=None")
        else:
            print(n, p.grad.norm().item())
        cnt+=1
        if cnt>=10: break
