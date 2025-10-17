import os
import torch
import yaml
from easydict import EasyDict
from flip_hf import CloningCLF
from engine_hf import MultiTaskEngine, create_shared_multitask_model

MODEL_PATH = "/content/drive/MyDrive/protein_multitask_outputs/single_task_training/CloningCLF_best_model.pth"
OUTPUT_DIR = os.path.dirname(MODEL_PATH)
DATA_PATH = "./data"
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_file="config_hf.yaml"):
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return EasyDict(yaml.safe_load(f))
    else:
        cfg = {
            'model': {
                'type': 'shared_lora',
                'model_name': 'Rostlab/prot_bert_bfd',
                'readout': 'pooler',
                'freeze_bert': True,
                'lora_rank': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1
            },
            'engine': {'batch_size': 8, 'num_worker': 1, 'log_interval': 50},
        }
        return EasyDict(cfg)

class SingleTaskWrapper(torch.nn.Module):
    def __init__(self, shared_model, task_id=0, task_type='binary_classification'):
        super().__init__()
        self.shared_model = shared_model
        self.task_id = task_id
        self.task_type = task_type

    def forward(self, batch):
        return self.shared_model(batch, task_id=self.task_id)

    def parameters(self):
        return self.shared_model.parameters()

if __name__ == "__main__":
    print(">>> Loading configuration and dataset...")
    cfg = load_config()

    dataset = CloningCLF(path=DATA_PATH)
    _, valid_set, test_set = dataset.split()
    print(f"Validation: {len(valid_set)} samples | Test: {len(test_set)} samples")

    print(">>> Creating model...")
    task_cfg = {
        "name": "CloningCLF",
        "type": "binary_classification",
        "num_labels": 1,
        "loss": "binary_cross_entropy"
    }
    shared_model = create_shared_multitask_model(tasks_config=[task_cfg], model_config=cfg.model)
    wrapped_model = SingleTaskWrapper(shared_model, task_id=0, task_type="binary_classification")

    print(f">>> Loading weights from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    wrapped_model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    wrapped_model.to(DEVICE)

    print(">>> Initializing engine for evaluation...")
    solver = MultiTaskEngine(
        tasks=[wrapped_model],
        train_sets=[[]],
        valid_sets=[valid_set],
        test_sets=[test_set],
        optimizer=None,
        scheduler=None,
        batch_size=BATCH_SIZE,
        num_worker=1,
        gradient_interval=1,
        log_interval=50
    )

    print(">>> Evaluating on validation set...")
    metrics_valid = solver.evaluate("valid")
    print(metrics_valid)

    print(">>> Evaluating on test set...")
    metrics_test = solver.evaluate("test")
    print(metrics_test)

    with open(os.path.join(OUTPUT_DIR, "CloningCLF_accuracy_evaluation.txt"), "w") as f:
        f.write("Validation metrics:\n")
        f.write(str(metrics_valid) + "\n\n")
        f.write("Test metrics:\n")
        f.write(str(metrics_test) + "\n")

    print("Evaluation complete! Results saved to:")
    print(os.path.join(OUTPUT_DIR, "CloningCLF_accuracy_evaluation.txt"))
