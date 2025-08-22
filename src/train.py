"""
FOST-PEFT Training Module
Implements the core FOST-PEFT training logic with orthogonal controllers, forecasters, and risk budgets.
"""

import os
import math
import time
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import geoopt
    _GEOOPT_AVAILABLE = True
except ImportError:
    _GEOOPT_AVAILABLE = False

try:
    from sklearn.random_projection import SparseRandomProjection
    from sklearn.isotonic import IsotonicRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False



class SeededSRP:
    """Seeded sparse random projection for compressed anchors."""
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0, dense_output: bool = True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seed = seed
        self.dense_output = dense_output
        if _SKLEARN_AVAILABLE:
            self._rp = SparseRandomProjection(n_components=out_dim, random_state=seed, dense_output=dense_output)
        else:
            rng = np.random.default_rng(seed)
            self._W = rng.standard_normal((in_dim, out_dim)).astype(np.float32) / math.sqrt(out_dim)

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if _SKLEARN_AVAILABLE:
            return self._rp.fit_transform(X)
        else:
            return X @ self._W


class AnchorMemory:
    """Privacy-lean anchor memory storing only compressed means and logits EMAs."""
    def __init__(self, in_dim: int, proj_dim: int = 64, seed: int = 0, dp_sigma: float = 0.0):
        self.rp = SeededSRP(in_dim, proj_dim, seed=seed)
        self.proj_dim = proj_dim
        self.dp_sigma = dp_sigma
        self.anchors: Dict[int, Dict[str, np.ndarray]] = {}

    def add_batch(self, feats: np.ndarray, logits: np.ndarray, ids: List[int]):
        Z = self.rp.project(feats)
        if self.dp_sigma > 0:
            Z = Z + np.random.normal(0, self.dp_sigma, size=Z.shape)
        for z, lg, k in zip(Z, logits, ids):
            if int(k) not in self.anchors:
                self.anchors[int(k)] = {
                    'mean': np.zeros(self.proj_dim, dtype=np.float32),
                    'logits': np.zeros_like(lg, dtype=np.float32),
                    'count': 0
                }
            a = self.anchors[int(k)]
            a['count'] += 1
            a['mean'] += (z - a['mean']) / float(a['count'])
            a['logits'] += (lg - a['logits']) / float(a['count'])

    def get_matrix(self) -> np.ndarray:
        if len(self.anchors) == 0:
            return np.zeros((0, self.proj_dim), dtype=np.float32)
        return np.stack([v['mean'] for v in self.anchors.values()], axis=0)


class BOFTController(nn.Module):
    """Block-orthogonal controller Q applied to input features."""
    def __init__(self, in_dim: int, block: int = 64, mode: str = 'stiefel'):
        super().__init__()
        self.in_dim = in_dim
        self.block = block
        self.blocks = nn.ParameterList()
        self.slices: List[slice] = []
        nb = (in_dim + block - 1) // block
        for b in range(nb):
            s = slice(b * block, min((b + 1) * block, in_dim))
            bs = s.stop - s.start
            if mode == 'stiefel' and _GEOOPT_AVAILABLE:
                man = geoopt.manifolds.stiefel.Stiefel()
                Qb = geoopt.ManifoldParameter(torch.eye(bs), manifold=man)
            else:
                Qb = nn.Parameter(torch.eye(bs))
            self.blocks.append(Qb)
            self.slices.append(s)
        self.mode = mode if _GEOOPT_AVAILABLE else 'soft'

    def apply_to_input(self, x: torch.Tensor) -> torch.Tensor:
        chunks = []
        for Qb, s in zip(self.blocks, self.slices):
            chunk = x[:, s] @ Qb
            chunks.append(chunk)
        return torch.cat(chunks, dim=1)

    def orth_penalty(self) -> torch.Tensor:
        if self.mode == 'stiefel' and _GEOOPT_AVAILABLE:
            return torch.tensor(0.0, device=self.blocks[0].device)
        pen = torch.tensor(0.0, device=self.blocks[0].device)
        for Qb in self.blocks:
            I = torch.eye(Qb.shape[0], device=Qb.device)
            pen = pen + torch.norm(Qb.T @ Qb - I, p='fro') ** 2
        return pen

    def rotation_energy(self) -> float:
        with torch.no_grad():
            e = 0.0
            for Qb in self.blocks:
                I = torch.eye(Qb.shape[0], device=Qb.device)
                e += torch.norm(Qb - I, p='fro').item()
        return e


class LoRALinear(nn.Module):
    """Standard LoRA linear layer."""
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r) if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.weight.T + self.bias
        if self.r <= 0:
            return base
        z = self.dropout(x @ self.lora_A.T)
        delta = z @ self.lora_B.T
        return base + self.scaling * delta


class FOSTLoRALinear(nn.Module):
    """FOST-PEFT enhanced LoRA layer with orthogonal controller and masking."""
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0, use_q: bool = True, block: int = 64, q_mode: str = 'stiefel'):
        super().__init__()
        self.lora = LoRALinear(in_features, out_features, r=r, alpha=alpha, dropout=dropout)
        self.r = r
        self.mask = torch.ones(r, dtype=torch.bool) if r > 0 else None
        self.use_q = use_q and (r > 0)
        self.qctrl = BOFTController(in_features, block=block, mode=q_mode) if self.use_q else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r <= 0:
            return self.lora(x)
        if self.use_q and self.qctrl is not None:
            x_rot = self.qctrl.apply_to_input(x)
        else:
            x_rot = x
        
        base = x @ self.lora.weight.T + self.lora.bias
        z = x_rot @ self.lora.lora_A.T
        delta = z @ self.lora.lora_B.T
        return base + self.lora.scaling * delta

    def set_mask(self, mask_bool: torch.Tensor):
        if self.r > 0:
            self.mask = mask_bool.to(self.lora.lora_A.device)

    def apply_mask_to_grads(self):
        if self.r <= 0 or self.mask is None:
            return
        if self.lora.lora_B.grad is not None:
            self.lora.lora_B.grad[:, ~self.mask] = 0.0
        if self.lora.lora_A.grad is not None:
            self.lora.lora_A.grad[~self.mask, :] = 0.0

    def orth_penalty(self) -> torch.Tensor:
        if not self.use_q or self.qctrl is None:
            return torch.tensor(0.0, device=self.lora.weight.device)
        return self.qctrl.orth_penalty()

    def rotation_energy(self) -> float:
        if not self.use_q or self.qctrl is None:
            return 0.0
        return self.qctrl.rotation_energy()


class LinearDiagForecaster:
    """Linearized forecaster for predicting forgetting risk."""
    def __init__(self, proj_dim: int, lr: float = 0.1, l2: float = 1e-2):
        self.c = np.zeros(proj_dim, dtype=np.float32)
        self.lr = lr
        self.l2 = l2
        self._buf_pred: List[float] = []
        self._buf_label: List[float] = []
        self._calib = IsotonicRegression(out_of_bounds='clip') if _SKLEARN_AVAILABLE else None
        self._fitted = False

    def predict_raw(self, u_proj: np.ndarray, anchors: np.ndarray) -> float:
        if anchors.shape[0] == 0:
            return 0.0
        vals = - (u_proj * self.c).reshape(1, -1) @ anchors.T
        return float(np.mean(vals))

    def update_ridge(self, u_proj: np.ndarray, anchors: np.ndarray, delta_m: float) -> float:
        pred = self.predict_raw(u_proj, anchors)
        if anchors.shape[0] == 0:
            return pred
        g = (pred - delta_m)
        mean_ua = np.mean(u_proj * anchors, axis=0)
        grad_c = g * (-mean_ua) + self.l2 * self.c
        self.c -= self.lr * grad_c.astype(np.float32)
        return pred

    def prob_and_ucb(self, raw_scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        raw = np.array(raw_scores)
        if not _SKLEARN_AVAILABLE or not self._fitted or self._calib is None:
            p = 1.0 / (1.0 + np.exp(-raw))
            return p, p
        p = self._calib.predict(raw)
        return p, p


class RiskBudget:
    """Risk budget controller with dual variable updates."""
    def __init__(self, B: float = 0.02, rho: float = 0.05):
        self.B = B
        self.rho = rho
        self.nu = 0.0
        self.ema_risk = 0.0

    def update(self, measured_risk: float):
        self.ema_risk = 0.9 * self.ema_risk + 0.1 * measured_risk
        self.nu = max(0.0, self.nu + self.rho * (self.ema_risk - self.B))

    def threshold(self) -> float:
        return self.B + 0.1 * self.nu


class FOSTModel(nn.Module):
    """Simple model with FOST-PEFT layers for testing."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, r: int = 8):
        super().__init__()
        self.layer1 = FOSTLoRALinear(input_dim, hidden_dim, r=r)
        self.layer2 = FOSTLoRALinear(hidden_dim, output_dim, r=r)
        self.relu = nn.ReLU()
        
        self.anchor_memory = AnchorMemory(hidden_dim, proj_dim=32)
        self.forecaster = LinearDiagForecaster(proj_dim=32)
        self.risk_budget = RiskBudget(B=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.layer1(x))
        out = self.layer2(h)
        return out

    def get_orth_penalty(self) -> torch.Tensor:
        return self.layer1.orth_penalty() + self.layer2.orth_penalty()


def train_fost_model(model: FOSTModel, dataloaders: List[DataLoader], 
                    n_epochs: int = 5, lr: float = 1e-3, device: str = 'cuda') -> Dict[str, List[float]]:
    """
    Train FOST-PEFT model on continual learning stream.
    
    Args:
        model: FOST model to train
        dataloaders: List of DataLoaders for each task
        n_epochs: Epochs per task
        lr: Learning rate
        device: Device to use
        
    Returns:
        Dictionary with training metrics
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {
        'task_losses': [],
        'task_accuracies': [],
        'rotation_energies': [],
        'risk_budgets': []
    }
    
    print(f"Training FOST-PEFT model on {len(dataloaders)} tasks...")
    
    for task_id, dataloader in enumerate(dataloaders):
        print(f"\n=== Task {task_id + 1}/{len(dataloaders)} ===")
        
        task_losses = []
        task_correct = 0
        task_total = 0
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (features, labels) in enumerate(dataloader):
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(features)
                task_loss = criterion(outputs, labels)
                
                orth_penalty = model.get_orth_penalty()
                total_loss = task_loss + 0.01 * orth_penalty
                
                total_loss.backward()
                
                if hasattr(model.layer1, 'apply_mask_to_grads'):
                    model.layer1.apply_mask_to_grads()
                if hasattr(model.layer2, 'apply_mask_to_grads'):
                    model.layer2.apply_mask_to_grads()
                
                optimizer.step()
                
                epoch_loss += total_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()
                
                model.risk_budget.update(0.01)  # Mock risk value
            
            epoch_acc = 100.0 * epoch_correct / epoch_total
            avg_loss = epoch_loss / len(dataloader)
            
            if epoch % 2 == 0:  # Print every 2 epochs
                print(f"  Epoch {epoch + 1}/{n_epochs}: Loss = {avg_loss:.4f}, Acc = {epoch_acc:.2f}%")
            
            task_losses.append(avg_loss)
            task_correct += epoch_correct
            task_total += epoch_total
        
        task_acc = 100.0 * task_correct / (task_total * n_epochs)
        avg_task_loss = np.mean(task_losses)
        rotation_energy = model.layer1.rotation_energy() + model.layer2.rotation_energy()
        
        metrics['task_losses'].append(avg_task_loss)
        metrics['task_accuracies'].append(task_acc)
        metrics['rotation_energies'].append(rotation_energy)
        metrics['risk_budgets'].append(model.risk_budget.nu)
        
        print(f"Task {task_id + 1} completed: Loss = {avg_task_loss:.4f}, Acc = {task_acc:.2f}%")
        print(f"  Rotation Energy = {rotation_energy:.4f}, Risk Budget ν = {model.risk_budget.nu:.4f}")
    
    print(f"\nTraining completed! Final average accuracy: {np.mean(metrics['task_accuracies']):.2f}%")
    return metrics


if __name__ == "__main__":
    print("Testing FOST-PEFT training components...")
    
    from preprocess import generate_synthetic_stream, create_dataloaders
    tasks = generate_synthetic_stream(n_tasks=3, samples_per_task=100, input_dim=64, n_classes=5)
    dataloaders = create_dataloaders(tasks, batch_size=16)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FOSTModel(input_dim=64, hidden_dim=32, output_dim=5, r=4)
    
    metrics = train_fost_model(model, dataloaders, n_epochs=2, lr=1e-3, device=device)
    
    print("Training test completed successfully!")
    print(f"Final metrics: {metrics}")
