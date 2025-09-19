import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np
from collections import defaultdict
import math
from protbert_hf import ProtBert
from torch.nn.utils.rnn import pad_sequence

class ProteinScaleExtractor:
    
    def __init__(self):
        # Common protein motifs for motif-level analysis
        self.functional_motifs = {
            'helix_start': ['EAEAK', 'AEAEK', 'KAAEK'],
            'helix_end': ['AEEAA', 'KEAAE'],
            'sheet_pattern': ['VIVIV', 'ILVIL', 'LILIV'],
            'turn_pattern': ['GPNG', 'PNGP', 'NGPN'],
            'catalytic_triad': ['HIS', 'CYS', 'ASP', 'SER'],
            'binding_site': ['DXDXD', 'GXGXXG'],
            'hydrophobic': ['LLVV', 'VVLL', 'LLLL'],
            'charged': ['DEKR', 'KRDE', 'EEKK']
        }
        
        # Amino acid properties for better motif encoding
        self.aa_properties = {
            'A': [0, 0, 0, 1],  # [charged, polar, hydrophobic, small]
            'R': [1, 1, 0, 0],  # [charged, polar, hydrophobic, small]
            'N': [0, 1, 0, 1],
            'D': [1, 1, 0, 1],
            'C': [0, 0, 1, 1],
            'Q': [0, 1, 0, 0],
            'E': [1, 1, 0, 0],
            'G': [0, 0, 0, 1],
            'H': [1, 1, 0, 1],
            'I': [0, 0, 1, 0],
            'L': [0, 0, 1, 0],
            'K': [1, 1, 0, 0],
            'M': [0, 0, 1, 0],
            'F': [0, 0, 1, 0],
            'P': [0, 0, 0, 1],
            'S': [0, 1, 0, 1],
            'T': [0, 1, 0, 1],
            'W': [0, 0, 1, 0],
            'Y': [0, 1, 1, 0],
            'V': [0, 0, 1, 1],
            'X': [0, 0, 0, 0]  # Unknown
        }
    
    def extract_amino_acid_features(self, sequence):
        features = []
        for aa in sequence:
            props = self.aa_properties.get(aa, [0, 0, 0, 0])
            features.append(props)
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_motifs(self, sequence, window_sizes=[3, 6, 9]):
        all_motifs = []
        all_positions = []
        
        # Fixed: Pad all motifs to consistent size (9 * 4 = 36 features max)
        max_motif_features = 36  # 9 amino acids * 4 properties each
        
        for window_size in window_sizes:
            for i in range(len(sequence) - window_size + 1):
                motif = sequence[i:i+window_size]
                # Convert motif to feature vector
                motif_features = []
                for aa in motif:
                    motif_features.extend(self.aa_properties.get(aa, [0, 0, 0, 0]))
                
                # FIXED: Pad motif_features to consistent length
                while len(motif_features) < max_motif_features:
                    motif_features.append(0.0)  # Pad with zeros
                
                # Truncate if somehow longer (shouldn't happen with our window sizes)
                motif_features = motif_features[:max_motif_features]
                
                all_motifs.append(motif_features)
                all_positions.append((i, i+window_size, window_size))
        
        return torch.tensor(all_motifs, dtype=torch.float32), all_positions
    
    def extract_domains(self, sequence, domain_size=50, overlap=10):
        domains = []
        positions = []
        
        start = 0
        while start < len(sequence):
            end = min(start + domain_size, len(sequence))
            domain = sequence[start:end]
            
            if len(domain) >= 20:  # Minimum meaningful domain size
                # Domain-level statistics
                aa_counts = defaultdict(int)
                for aa in domain:
                    aa_counts[aa] += 1
                
                # Convert to feature vector (20 AA frequencies + length)
                domain_features = []
                for aa in 'ACDEFGHIKLMNPQRSTVWY':
                    domain_features.append(aa_counts[aa] / len(domain))
                domain_features.append(len(domain) / 500.0)  # Normalized length
                
                # Add biochemical properties summary
                total_charged = sum(aa_counts[aa] for aa in 'DEKR')
                total_hydrophobic = sum(aa_counts[aa] for aa in 'ILVFWYC')
                total_polar = sum(aa_counts[aa] for aa in 'STYNQH')
                
                domain_features.extend([
                    total_charged / len(domain),
                    total_hydrophobic / len(domain),
                    total_polar / len(domain)
                ])
                
                domains.append(domain_features)
                positions.append((start, end))
            
            start += domain_size - overlap
        
        if domains:
            return torch.tensor(domains, dtype=torch.float32), positions
        else:
            # Return dummy domain if sequence too short
            dummy_domain = [0.0] * 24  # 20 AA + 1 length + 3 properties
            return torch.tensor([dummy_domain], dtype=torch.float32), [(0, len(sequence))]


class MultiScaleEncoder(nn.Module):
  
    def __init__(self, protbert_model, hidden_dim=None):
        super().__init__()
        
        self.protbert = protbert_model
        self.hidden_dim = hidden_dim or protbert_model.output_dim
        self.scale_extractor = ProteinScaleExtractor()
        
        # Scale-specific encoders
        # 1. Amino acid level: use ProtBert directly
        
        # 2. Motif level encoder
        # AFTER (correct):
        self.motif_encoder = nn.Sequential(
            nn.Linear(36, self.hidden_dim // 4),  # <-- self.hidden_dim
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 2)  # <-- self.hidden_dim
        )

        self.domain_encoder = nn.Sequential(
            nn.Linear(24, self.hidden_dim // 2),  # <-- self.hidden_dim
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2)  # <-- self.hidden_dim
        )

        self.scale_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,  # <-- self.hidden_dim
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.motif_proj = nn.Linear(self.hidden_dim // 2, self.hidden_dim)  # <-- self.hidden_dim
        self.domain_proj = nn.Linear(self.hidden_dim // 2, self.hidden_dim)  # <-- self.hidden_dim

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),  # <-- self.hidden_dim
            nn.LayerNorm(self.hidden_dim),  # <-- self.hidden_dim
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Scale-specific weights (learnable)
        self.scale_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, batch):
        print(f"[DEBUG] MultiScaleEncoder.forward called")
        print(f"[DEBUG] batch type: {type(batch)}")
        print(f"[DEBUG] batch content: {batch}")

        sequences = batch.get('sequence', [])
        device = next(self.parameters()).device
        
        # 1. Amino acid level (sequence level from ProtBert)
        protbert_outputs = self.protbert(batch)
        aa_features = protbert_outputs["graph_feature"]  # [batch_size, hidden_dim]
        
        # 2. Motif level
        batch_motif_features = []
        for seq in sequences:
            motif_features, _ = self.scale_extractor.extract_motifs(seq)
            if motif_features.size(0) > 0:
                # Pad to consistent size
                max_features = 36
                if motif_features.size(1) < max_features:
                    padding = torch.zeros(motif_features.size(0), max_features - motif_features.size(1))
                    motif_features = torch.cat([motif_features, padding], dim=1)
                else:
                    motif_features = motif_features[:, :max_features]
                
                # Pool motifs to get sequence-level representation
                motif_repr = self.motif_encoder(motif_features.to(device))
                motif_repr = motif_repr.mean(dim=0)  # Pool over motifs
            else:
                motif_repr = torch.zeros(self.hidden_dim // 2, device=device)
            
            batch_motif_features.append(motif_repr)
        
        motif_features = torch.stack(batch_motif_features)
        motif_features = self.motif_proj(motif_features)
        
        # 3. Domain level
        batch_domain_features = []
        for seq in sequences:
            domain_features, _ = self.scale_extractor.extract_domains(seq)
            domain_repr = self.domain_encoder(domain_features.to(device))
            domain_repr = domain_repr.mean(dim=0)  # Pool over domains
            batch_domain_features.append(domain_repr)
        
        domain_features = torch.stack(batch_domain_features)
        domain_features = self.domain_proj(domain_features)
        
        # Cross-scale attention
        scales = torch.stack([aa_features, motif_features, domain_features], dim=1)  # [batch, 3, hidden]
        attended_scales, attention_weights = self.scale_attention(scales, scales, scales)
        
        # Weighted combination
        scale_weights = F.softmax(self.scale_weights, dim=0)
        weighted_scales = attended_scales * scale_weights.view(1, 3, 1)
        
        # Fusion
        fused_input = weighted_scales.flatten(start_dim=1)  # [batch, 3*hidden]
        fused_features = self.fusion_layer(fused_input)
        
        return {
            "graph_feature": fused_features,           # Fused multi-scale
            "aa_features": aa_features,               # Amino acid level
            "motif_features": motif_features,         # Motif level  
            "domain_features": domain_features,       # Domain level
            "attention_weights": attention_weights,   # For interpretability
            "scale_weights": scale_weights,          # Learned scale importance
            "residue_feature": protbert_outputs.get("residue_feature"),  # For token tasks
            "attention_mask": protbert_outputs.get("attention_mask")
        }


class ScaleAdaptiveTaskHeads(nn.Module):
    
    def __init__(self, hidden_dim, tasks_config):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tasks_config = tasks_config
        
        # Define which scales are most relevant for each task
        self.task_scale_preferences = {
            0: ['aa', 'motif'],      # Thermostability: local + motif patterns
            1: ['aa', 'motif'],      # Secondary structure: local patterns  
            2: ['motif', 'domain']   # Binding affinity: binding motifs + domain
        }
        
        # Scale-specific projections for each task
        self.task_scale_projections = nn.ModuleDict()
        self.task_heads = nn.ModuleDict()
        
        for task_id, task_config in enumerate(tasks_config):
            task_name = f"task_{task_id}"
            preferred_scales = self.task_scale_preferences.get(task_id, ['aa', 'motif', 'domain'])
            
            # Scale combination layer
            input_dim = len(preferred_scales) * hidden_dim
            self.task_scale_projections[task_name] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Task-specific head
            if task_config['type'] == 'token_classification':
                # Use residue-level features
                self.task_heads[task_name] = nn.Linear(hidden_dim, task_config['num_labels'])
            else:
                # Use fused sequence-level features
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, task_config['num_labels'])
                )
    
    def forward(self, multiscale_outputs, task_id):
        task_name = f"task_{task_id}"
        preferred_scales = self.task_scale_preferences.get(task_id, ['aa', 'motif', 'domain'])
        
        # Combine preferred scales
        scale_features = []
        for scale in preferred_scales:
            if scale == 'aa':
                scale_features.append(multiscale_outputs["aa_features"])
            elif scale == 'motif':
                scale_features.append(multiscale_outputs["motif_features"])
            elif scale == 'domain':
                scale_features.append(multiscale_outputs["domain_features"])
        
        combined_features = torch.cat(scale_features, dim=-1)
        task_features = self.task_scale_projections[task_name](combined_features)
        
        # Handle token classification differently
        if self.tasks_config[task_id]['type'] == 'token_classification':
            residue_features = multiscale_outputs.get("residue_feature")
            if residue_features is not None:
                # Project task features to residue level
                batch_size, seq_len, _ = residue_features.shape
                task_features_expanded = task_features.unsqueeze(1).expand(-1, seq_len, -1)
                
                # Combine with residue features
                combined = residue_features + task_features_expanded
                logits = self.task_heads[task_name](combined)
            else:
                # Fallback: expand task features to sequence length (simplified)
                logits = self.task_heads[task_name](task_features).unsqueeze(1)
        else:
            logits = self.task_heads[task_name](task_features)
        
        return logits


class MultiScaleMultiTaskModel(nn.Module):

    def __init__(self, protbert_model, tasks_config, diversity_weight=0.1):
        super().__init__()
        
        self.multiscale_encoder = MultiScaleEncoder(protbert_model)
        self.task_heads = ScaleAdaptiveTaskHeads(protbert_model.output_dim, tasks_config)
        self.diversity_weight = diversity_weight
        self.tasks_config = tasks_config
        
    def forward(self, batch, task_id):
        # Multi-scale encoding
        multiscale_outputs = self.multiscale_encoder(batch)
        
        # Task-specific prediction using appropriate scales
        logits = self.task_heads(multiscale_outputs, task_id)
        
        # Scale diversity loss to encourage different scales to capture different info
        diversity_loss = self.compute_diversity_loss(multiscale_outputs)
        
        return {
            "logits": logits,
            "multiscale_outputs": multiscale_outputs,
            "diversity_loss": diversity_loss
        }
    
    def compute_diversity_loss(self, multiscale_outputs):
        aa_feat = multiscale_outputs["aa_features"]
        motif_feat = multiscale_outputs["motif_features"] 
        domain_feat = multiscale_outputs["domain_features"]
        
        # Compute pairwise similarities
        aa_motif_sim = F.cosine_similarity(aa_feat, motif_feat, dim=1).mean()
        aa_domain_sim = F.cosine_similarity(aa_feat, domain_feat, dim=1).mean()
        motif_domain_sim = F.cosine_similarity(motif_feat, domain_feat, dim=1).mean()
        
        # Diversity loss (we want scales to be different, so penalize high similarity)
        diversity_loss = (aa_motif_sim + aa_domain_sim + motif_domain_sim) / 3
        
        return self.diversity_weight * diversity_loss

class MultiScaleModelsWrapper(nn.Module):

    def __init__(self, multiscale_model, task_names):
        super().__init__()
        self.multiscale_model = multiscale_model
        self.names = task_names
        
    def forward(self, batches):
        all_loss = []
        all_metric = defaultdict(float)
        total_diversity_loss = 0
        
        for task_id, batch in enumerate(batches):
            outputs = self.multiscale_model(batch, task_id)
            
            # Compute task loss
            #loss = self.compute_task_loss(outputs["logits"], batch, task_id)
            loss = self.compute_default_loss(outputs, batch, task_id)

            # Add diversity loss
            diversity_loss = outputs.get("diversity_loss", 0)
            total_loss = loss + diversity_loss
            total_diversity_loss += diversity_loss
            
            # Compute metrics
            #metrics = self.compute_task_metrics(outputs["logits"], batch, task_id)
            metrics = self.compute_default_metrics(outputs, batch, task_id)

            for k, v in metrics.items():
                name = self.names[task_id] + " " + k
                if task_id == 0:
                    name = "Center - " + name
                all_metric[name] = v
            
            all_loss.append(total_loss)
        
        # Add average diversity loss to metrics
        all_metric["diversity_loss"] = total_diversity_loss / len(batches)
        
        all_loss = torch.stack(all_loss)
        return all_loss, all_metric
    
    def compute_default_loss(self, outputs, batch, task_id):
        logits = outputs["logits"]  # raw outputs of the final layer before softmax/sigmoid

        # Extract targets
        if isinstance(batch, dict) and 'targets' in batch:
            targets = batch['targets']
            if isinstance(targets, dict):
                print("Targets is a dict, using the first key for loss computation.")
                print("Available keys:", list(targets.keys()))
                target_key = list(targets.keys())[0]
                target = targets[target_key]  # pick first if multiple
                print("Using target key:", target_key)
                print("Targets example:", target if isinstance(target, (int, float, list)) else str(target)[:100])
            else:
                print("Targets is not a dict, using it directly for loss computation.")
                print("Targets type:", type(targets))
                print("Targets example:", targets if isinstance(targets, (int, float, list)) else str(targets)[:100])
                target = targets
        else:
            raise ValueError("Cannot find targets in batch")

        print("\n[DEBUG] ====== New Batch ======")
        if "sequence" in batch:
            print("Example sequence:", batch["sequence"][0][:50], "...")
        print("Targets (first item):", target[0] if isinstance(target, (list, torch.Tensor)) else target)
        print("Logits shape:", logits.shape)

        # Determine task type
        #task_type = task_type or getattr(self, 'task_type', None) or batch.get('task_type', 'regression')
        task_config = self.multiscale_model.tasks_config[task_id]
        task_type = task_config['type']

        # Special case: force Task_2 to multi-class classification
        if hasattr(self, 'names') and "Task_2" in self.names:
            task_type = "multi_class"
        print("[DEBUG] task_type detected:", task_type)

        # --- Token-level classification (sequence labeling) ---
        if (isinstance(target, list) and len(target) > 0 and isinstance(target[0], list)) or \
        (isinstance(target, torch.Tensor) and target.dim() == 2):

            if isinstance(target, list):
                tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
                target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(logits.device)
            else:
                target_tensor = target.to(logits.device)

            multiscale_outputs = outputs.get("multiscale_outputs", {})
            mask = multiscale_outputs.get("attention_mask")

            if mask is None:
                print("Warning: attention_mask not found, assuming all tokens are valid.")
                mask = torch.ones((target_tensor.size(0), logits.size(1)), dtype=torch.long, device=logits.device)
            else:
                mask = mask.to(logits.device)

            L_mask = mask.size(1)
            if target_tensor.size(1) > L_mask:
                target_tensor = target_tensor[:, :L_mask]
            elif target_tensor.size(1) < L_mask:
                pad = torch.full((target_tensor.size(0), L_mask - target_tensor.size(1)),
                                fill_value=-100, device=logits.device, dtype=target_tensor.dtype)
                target_tensor = torch.cat([target_tensor, pad], dim=1)

            active = mask.reshape(-1) == 1
            active_logits = logits.reshape(-1, logits.size(-1))[active]
            active_labels = target_tensor.reshape(-1)[active]

            if active_logits.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            return F.cross_entropy(active_logits, active_labels)  # token-level classification loss

        # --- Sequence-level tasks ---
        t = torch.tensor(target, device=logits.device, dtype=torch.float)

        if task_type == 'binary_classification':
            logits_ = logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            return F.binary_cross_entropy_with_logits(logits_, t.float())

        elif task_type == 'regression':
            if logits.numel() == t.numel():
                logits_flat = logits.view_as(t)
            elif logits.size(-1) == 1:
                logits_flat = logits.squeeze(-1)
            else:
                logits_flat = logits.mean(dim=1)
            return F.mse_loss(logits_flat, t.float())

        else:  # multi-class
            if logits.dim() == 1 or logits.size(-1) == 1:
                if logits.numel() == t.numel():
                    logits_flat = logits.view_as(t)
                else:
                    logits_flat = logits.squeeze(-1)
                return F.mse_loss(logits_flat, t.float())
            else:
                if t.dtype != torch.long:
                    t = t.long()
                t = torch.clamp(t, 0, logits.size(-1)-1)
                return F.cross_entropy(logits, t)

    def compute_default_metrics(self, outputs, batch, task_id):
        logits = outputs["logits"]
        print("[DEBUG] logits shape:", logits.shape)
        print("[DEBUG] batch keys:", batch.keys())

        # Extract targets
        if isinstance(batch, dict) and 'targets' in batch:
            targets = batch['targets']
            if isinstance(targets, dict):
                print("Targets is a dict, using the first key for metric computation.")
                target_key = list(targets.keys())[0]
                target = targets[target_key]
            else:
                print("Targets is not a dict, using it directly for metric computation.")
                target = targets
        else:
            raise ValueError("Cannot find targets in batch")

        print("[DEBUG] raw targets type:", type(target))

        # Determine task type
        #task_type = task_type or getattr(self, 'task_type', None) or batch.get('task_type', 'regression')
        task_config = self.multiscale_model.tasks_config[task_id]
        task_type = task_config['type']

        # Force Task_2 as multi-class
        if hasattr(self, 'names') and "Task_2" in self.names:
            task_type = "multi_class"
        print("[DEBUG] task_type detected:", task_type)

        # --- Token-level classification ---
        if (isinstance(target, list) and len(target) > 0 and isinstance(target[0], list)) or \
        (isinstance(target, torch.Tensor) and target.dim() == 2):

            if isinstance(target, list):
                tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in target]
                target_tensor = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100).to(logits.device)
            else:
                target_tensor = target.to(logits.device)

            multiscale_outputs = outputs.get("multiscale_outputs", {})
            mask = multiscale_outputs.get("attention_mask")
            
            if mask is None:
                raise ValueError("Token-level task requires attention_mask")
            mask = mask.to(logits.device)

            L_mask = mask.size(1)
            if target_tensor.size(1) > L_mask:
                target_tensor = target_tensor[:, :L_mask]
            elif target_tensor.size(1) < L_mask:
                pad = torch.full((target_tensor.size(0), L_mask - target_tensor.size(1)),
                                fill_value=-100, device=logits.device, dtype=target_tensor.dtype)
                target_tensor = torch.cat([target_tensor, pad], dim=1)

            pred = logits.argmax(dim=-1)
            active = mask.reshape(-1) == 1
            active_preds = pred.reshape(-1)[active]
            active_labels = target_tensor.reshape(-1)[active]

            acc = (active_preds == active_labels).float().mean().item() if active_preds.numel() > 0 else 0.0
            print("[DEBUG] Token-level task")
            print("[DEBUG] target_tensor shape:", target_tensor.shape)
            print("[DEBUG] mask shape:", mask.shape)
            print("[DEBUG] pred shape:", pred.shape)
            print("[DEBUG] active_preds size:", active_preds.size())
            print("[DEBUG] active_labels size:", active_labels.size())
            return {"accuracy": acc}

        # --- Sequence-level tasks ---
        t = torch.tensor(target, device=logits.device, dtype=torch.float)

        if task_type == 'binary_classification':
            logits_ = logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            pred = (torch.sigmoid(logits_) > 0.5).long()
            print("[DEBUG] Binary classification")
            return {"accuracy": (pred == t.long()).float().mean().item()}

        elif task_type == 'regression':
            if logits.numel() == t.numel():
                logits_flat = logits.view_as(t)
            elif logits.size(-1) == 1:
                logits_flat = logits.squeeze(-1)
            else:
                logits_flat = logits.mean(dim=1)
            mse = F.mse_loss(logits_flat, t.float()).item()
            print("[DEBUG] Regression")
            return {"mse": mse}

        else:  # multi-class classification
            if t.dtype != torch.long:
                t = t.long()
            t = torch.clamp(t, 0, logits.size(-1)-1)
            pred = logits.argmax(dim=-1)
            acc = (pred == t).float().mean().item()
            print("[DEBUG] Multi-class classification")
            return {"accuracy": acc}
    
    def __getitem__(self, idx):
        
        class TaskModel:
            def __init__(self, wrapper, task_id):
                self.wrapper = wrapper
                self.task_id = task_id
                
            def __call__(self, batch):
                print(f"[DEBUG] TaskModel called for task {self.task_id}")
                print(f"[DEBUG] batch type in TaskModel: {type(batch)}")
                return self.wrapper.multiscale_model(batch, self.task_id)
            
            def forward(self, batch):
                return self(batch)
                
            def to(self, device):
                return self
                
            def eval(self):
                self.wrapper.multiscale_model.eval()
                return self
                
            def train(self):
                self.wrapper.multiscale_model.train()
                return self
                
            def parameters(self):
                return self.wrapper.multiscale_model.parameters()
        
        return TaskModel(self, idx)

def create_multiscale_shared_model(tasks_config, model_config):    
    # Create base ProtBert
    protbert = ProtBert(
        model_name=model_config['model_name'],
        readout=model_config['readout'],
        freeze_bert=model_config.get('freeze_bert', False)
    )
    
    # Create multi-scale model
    multiscale_model = MultiScaleMultiTaskModel(
        protbert_model=protbert,
        tasks_config=tasks_config,
        diversity_weight=model_config.get('diversity_weight', 0.05)
    )
    
    return multiscale_model