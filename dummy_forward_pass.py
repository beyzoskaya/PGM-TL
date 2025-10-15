import torch
from engine_hf import SharedBackboneMultiTaskModel  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Initializing dummy forward pass test...")

model = SharedBackboneMultiTaskModel()
model.to(device)
model.eval()

sample = {
    'sequence': 'SNAMFCYQ...',
    'targets': {'label': [0,1,2]},
    'graph': {
        'residue_type': torch.randint(0, 20, (100,)),
        'num_residues': torch.tensor([100]),
        'batch_size': 1
    }
}

print("[INFO] Sample keys:", sample.keys())
print("[INFO] Graph keys:", sample['graph'].keys())

# Instead of unpacking, pass the graph directly
graph_input = sample['graph']  # pass as single argument
# If the model expects a tensor batch, make it a batch of 1
if isinstance(graph_input['residue_type'], torch.Tensor):
    graph_input = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v
                   for k, v in graph_input.items()}

with torch.no_grad():
    try:
        outputs = model(graph_input)  # pass as a single object
        print("[INFO] Forward pass successful!")
        print("Output type:", type(outputs))
    except Exception as e:
        print("[ERROR] Forward pass failed:", e)
