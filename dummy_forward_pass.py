import torch
from engine_hf import SharedBackboneMultiTaskModel  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Initializing dummy forward pass test...")

model = SharedBackboneMultiTaskModel()
model.to(device)
model.eval()

sample = {
    'sequence': 'SNAMFCYQ...',
    'targets': {'label': [0,1,2]},  # dummy targets
    'graph': {
        'residue_type': torch.randint(0, 20, (100,)),  # example: 100 residues
        'num_residues': torch.tensor([100]),
        'batch_size': 1
    }
}

print("[INFO] Sample keys:", sample.keys())
print("[INFO] Graph keys:", sample['graph'].keys())

graph_inputs = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v
                for k, v in sample['graph'].items()}

print("[INFO] Graph inputs prepared:")
for k, v in graph_inputs.items():
    print(f"  {k}: {v.shape if torch.is_tensor(v) else v}")

with torch.no_grad():
    try:
        outputs = model(**graph_inputs)
        print("[INFO] Forward pass successful!")
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                print(f"  Output {k}: {v.shape if torch.is_tensor(v) else v}")
        else:
            print("[INFO] Model output:", outputs)
    except Exception as e:
        print("[ERROR] Forward pass failed:", e)
