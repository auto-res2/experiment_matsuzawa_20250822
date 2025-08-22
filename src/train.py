import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import pickle

from preprocess import set_seed


class MLPClassifier(nn.Module):
    """Multi-layer perceptron for binary classification."""
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 2, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class TemperatureScaler(nn.Module):
    """Temperature scaling for calibration."""
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))
    
    def forward(self, logits):
        T = torch.exp(self.log_temp) + 1e-6
        return logits / T
    
    @torch.no_grad()
    def predict_proba(self, logits):
        scaled = self.forward(logits)
        return F.softmax(scaled, dim=-1)


class TorchProbaWrapper:
    """Wrapper for PyTorch model to provide sklearn-like interface."""
    def __init__(self, model: MLPClassifier, scaler: TemperatureScaler, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.scaler = scaler.to(device).eval()
        self.device = device
    
    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        logits = self.model(X)
        probs = self.scaler.predict_proba(logits)
        return probs.cpu().numpy()


def train_mlp_with_temp(train_X: np.ndarray, train_y: np.ndarray,
                        val_X: np.ndarray, val_y: np.ndarray,
                        in_dim: int, epochs: int = 25, lr: float = 1e-3,
                        device: str = "cpu") -> Tuple[MLPClassifier, TemperatureScaler, List[float]]:
    """Train MLP classifier with temperature scaling."""
    model = MLPClassifier(in_dim)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    Xtr = torch.tensor(train_X, dtype=torch.float32, device=device)
    ytr = torch.tensor(train_y, dtype=torch.long, device=device)
    Xva = torch.tensor(val_X, dtype=torch.float32, device=device)
    yva = torch.tensor(val_y, dtype=torch.long, device=device)
    
    best_state = None
    best_val = float("inf")
    losses = []
    
    print("Training MLP classifier...")
    for ep in range(epochs):
        model.train()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        with torch.no_grad():
            model.eval()
            val_loss = loss_fn(model(Xva), yva).item()
        
        losses.append(float(val_loss))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        if ep % 5 == 0:
            print(f"Epoch {ep}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)

    print("Performing temperature scaling...")
    scaler = TemperatureScaler().to(device)
    scaler_opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)
    
    def closure():
        scaler_opt.zero_grad()
        with torch.no_grad():
            logits = model(Xva)
        scaled = scaler(logits)
        loss = loss_fn(scaled, yva)
        loss.backward()
        return loss
    
    try:
        scaler_opt.step(closure)
    except Exception as e:
        print(f"Temperature scaling failed: {e}")
    
    return model, scaler, losses


class PartitionedQuantileCI:
    """Conformal prediction with partitioned quantile residuals."""
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.global_q = 0.25
        self.partition_q: Dict[Any, float] = {}
    
    def fit(self, p_hat: np.ndarray, y: np.ndarray, keys: List[Any] = None):
        """Fit conformal predictor on calibration data."""
        res = np.abs(y.astype(float) - p_hat.astype(float))
        if keys is None:
            keys = ["__global__"] * len(res)
        
        buckets: Dict[Any, List[float]] = {}
        for r, k in zip(res, keys):
            buckets.setdefault(k, []).append(float(r))
        
        self.partition_q = {}
        all_res = []
        for k, rs in buckets.items():
            rs_sorted = sorted(rs)
            q_idx = max(0, int(np.ceil((1 - self.alpha) * len(rs_sorted)) - 1))
            q = rs_sorted[q_idx]
            self.partition_q[k] = q
            all_res.extend(rs)
        
        if all_res:
            all_res_sorted = sorted(all_res)
            q_idx = max(0, int(np.ceil((1 - self.alpha) * len(all_res_sorted)) - 1))
            self.global_q = all_res_sorted[q_idx]
        else:
            self.global_q = 0.25
    
    def interval(self, p: float, key: Any = None) -> Tuple[float, float]:
        """Get confidence interval for prediction."""
        q = self.partition_q.get(key, self.global_q)
        lo = max(0.0, p - q)
        hi = min(1.0, p + q)
        return (lo, hi)


def train_models():
    """Main training function."""
    print("Starting model training...")
    set_seed(42)
    
    data_dir = "data"
    features = np.load(os.path.join(data_dir, "features.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    
    print(f"Loaded {len(features)} samples with {features.shape[1]} features")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.4, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, scaler, losses = train_mlp_with_temp(
        X_train, y_train, X_val, y_val, 
        in_dim=features.shape[1], epochs=30, device=device
    )
    
    wrapper = TorchProbaWrapper(model, scaler, device)
    
    test_probs = wrapper.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    
    print("\nTest Set Performance:")
    print(classification_report(y_test, test_preds))
    
    print("\nTraining conformal predictor...")
    val_probs = wrapper.predict_proba(X_val)
    conformal = PartitionedQuantileCI(alpha=0.1)
    conformal.fit(val_probs[:, 1], y_val)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(models_dir, "mlp_model.pth"))
    torch.save(scaler.state_dict(), os.path.join(models_dir, "temperature_scaler.pth"))
    
    with open(os.path.join(models_dir, "conformal_predictor.pkl"), "wb") as f:
        pickle.dump(conformal, f)
    
    training_info = {
        "feature_dim": features.shape[1],
        "device": device,
        "losses": losses,
        "test_accuracy": np.mean(test_preds == y_test),
        "test_probs": test_probs.tolist(),
        "test_labels": y_test.tolist()
    }
    
    with open(os.path.join(models_dir, "training_info.pkl"), "wb") as f:
        pickle.dump(training_info, f)
    
    print(f"Models saved to {models_dir}/")
    print("Training completed successfully!")
    
    return wrapper, conformal, training_info


if __name__ == "__main__":
    train_models()
