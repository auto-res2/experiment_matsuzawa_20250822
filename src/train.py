import os
import sys
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Ensure reproducible plots and PDF output quality
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def set_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# ------------------------------
# Synthetic manifold-like dataset
# ------------------------------

class SyntheticManifoldImages(Dataset):
    """
    Synthetic grayscale images (1x16x16) with two patterns:
      - 'blobs': a Gaussian blob at (u, v) with small pixel noise.
      - 'stripes': sinusoidal stripes with freq/phase from latent.
    Manifold latent is 2D (u,v) in [-1,1]^2. Labels depend on sign(u).
    """
    def __init__(self, n: int = 2000, pattern: str = 'blobs', normal_noise: float = 0.05, seed: int = 0):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.n = n
        self.pattern = pattern
        self.normal_noise = normal_noise
        self.size = 16
        # latent coordinates
        self.z = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
        self.y = (self.z[:, 0] > 0).astype(np.int64)
        self.X = np.zeros((n, 1, self.size, self.size), dtype=np.float32)
        for i in range(n):
            if pattern == 'blobs':
                self.X[i, 0] = self._gen_blob(self.z[i])
            elif pattern == 'stripes':
                self.X[i, 0] = self._gen_stripes(self.z[i])
            else:
                raise ValueError(f'Unknown pattern: {pattern}')
            self.X[i, 0] += rng.normal(0, normal_noise, size=(self.size, self.size)).astype(np.float32)
        self.X = np.clip(self.X, 0.0, 1.0)

    def _gen_blob(self, z):
        u, v = float(z[0]), float(z[1])
        cx = (u + 1) / 2 * (self.size - 1)
        cy = (v + 1) / 2 * (self.size - 1)
        xs = np.arange(self.size)
        ys = np.arange(self.size)
        X, Y = np.meshgrid(xs, ys)
        sigma = 2.0
        img = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        return img

    def _gen_stripes(self, z):
        u, v = float(z[0]), float(z[1])
        X = np.arange(self.size)
        Y = np.arange(self.size)
        XX, YY = np.meshgrid(X, Y)
        freq = 1.5 + 1.0 * u
        phase = np.pi * v
        img = 0.5 * (1 + np.sin(2 * np.pi * (XX / self.size) * freq + phase)).astype(np.float32)
        return img

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# ------------------------------
# Model
# ------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ------------------------------
# Geometry helpers
# ------------------------------

def proj_T(v: torch.Tensor, BT: torch.Tensor) -> torch.Tensor:
    if v.dim() == 1:
        return BT @ (BT.t() @ v)
    return (v @ BT) @ BT.t()


def whiten_x(x_flat: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float) -> torch.Tensor:
    PT_x = proj_T(x_flat, BT)
    PN_x = x_flat - PT_x
    return PT_x / sigma_T + PN_x / sigma_N


def dewhiten_delta_y(delta_y: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float) -> torch.Tensor:
    PT_y = proj_T(delta_y, BT)
    PN_y = delta_y - PT_y
    return PT_y * sigma_T + PN_y * sigma_N


# ------------------------------
# TaReS penalty
# ------------------------------

def tares_penalty(model: nn.Module, x: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float, K: int = 8) -> torch.Tensor:
    B = x.size(0)
    device = x.device
    x_flat = x.view(B, -1)
    y0 = whiten_x(x_flat, BT, sigma_T, sigma_N)
    y0_img = y0.view_as(x)
    eps = torch.randn(K // 2, *x.shape, device=device)
    eps = torch.cat([eps, -eps], dim=0)
    logits_acc = 0.0
    for e in torch.split(eps, 16, dim=0):
        logits_acc = logits_acc + model(y0_img + e)
    logits = logits_acc / (K)
    probs = F.softmax(logits, dim=1)
    y_pred = probs.argmax(dim=1)
    sel = logits[torch.arange(B), y_pred].sum()
    grad = torch.autograd.grad(sel, y0_img, retain_graph=False, create_graph=False)[0]
    grad_flat = grad.view(B, -1)
    gT = proj_T(grad_flat, BT)
    gN = grad_flat - gT
    return (gN.norm(dim=1) ** 2).mean()


# ------------------------------
# Training loop
# ------------------------------

def plot_training_curves(history: Dict[str, List[float]], condition: str, outdir: str):
    ensure_dir(outdir)
    # loss
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'training_loss_{condition}.pdf'))
    plt.close()
    # accuracy
    plt.figure()
    plt.plot(history['val_acc'], label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'accuracy_{condition}.pdf'))
    plt.close()


def train_model(train_loader: DataLoader, val_loader: DataLoader, n_classes: int, device: str,
                epochs: int = 5, lr: float = 1e-3, tares: bool = False, BT: torch.Tensor = None,
                sigma_T: float = 0.25, sigma_N: float = 0.15, lambda_R: float = 5e-4) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model = SimpleCNN(n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_acc': []}
    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if tares:
                assert BT is not None
                pen = tares_penalty(model, xb, BT, sigma_T, sigma_N, K=8)
                loss = loss + lambda_R * pen
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += xb.size(0)
        val_acc = correct / max(total, 1)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        print(f"[Train] epoch={ep+1} loss={train_loss:.4f} val_acc={val_acc:.4f}")
    return model, history


# ------------------------------
# Orchestration
# ------------------------------

def run(config_path: str = 'config/config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seeds(cfg.get('seed', 42))

    device = 'cuda' if (cfg.get('device', 'auto') == 'cuda' and torch.cuda.is_available()) else 'cpu'

    # Paths
    images_dir = cfg['paths'].get('images_dir', '.research/iteration1/images')
    models_dir = cfg['paths'].get('models_dir', 'models')
    data_dir = cfg['paths'].get('data_dir', 'data')
    ensure_dir(images_dir)
    ensure_dir(models_dir)
    ensure_dir(data_dir)

    # Data config
    pattern = cfg['data'].get('pattern', 'blobs')
    normal_noise = float(cfg['data'].get('normal_noise', 0.05))
    n_train = int(cfg['data'].get('n_train', 1000))
    n_val = int(cfg['data'].get('n_val', 200))

    # Geometry config
    k = int(cfg['geometry'].get('k', 4))
    sigma_T = float(cfg['geometry'].get('sigma_T', 0.25))
    sigma_N = float(cfg['geometry'].get('sigma_N', 0.15))

    # Load BT produced by preprocess
    bt_path = os.path.join(data_dir, f'BT_k{k}.pt')
    if not os.path.exists(bt_path):
        raise FileNotFoundError(f"BT file not found at {bt_path}. Run preprocess first.")
    BT = torch.load(bt_path, map_location=device)

    # Training config
    epochs = int(cfg['train'].get('epochs', 5))
    batch_size = int(cfg['train'].get('batch_size', 64))
    lr = float(cfg['train'].get('lr', 1e-3))
    lambda_R = float(cfg['train'].get('lambda_R', 5e-4))

    # Datasets and loaders
    train_ds = SyntheticManifoldImages(n=n_train, pattern=pattern, normal_noise=normal_noise, seed=0)
    val_ds = SyntheticManifoldImages(n=n_val, pattern=pattern, normal_noise=normal_noise, seed=10)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(64, batch_size), shuffle=False)

    # Train baseline
    print("\n== Training baseline model ==")
    model_base, hist_base = train_model(train_loader, val_loader, n_classes=2, device=device, epochs=epochs, lr=lr, tares=False)
    plot_training_curves(hist_base, condition='baseline', outdir=images_dir)
    base_path = os.path.join(models_dir, 'simplecnn_baseline.pt')
    torch.save(model_base.state_dict(), base_path)

    # Train TaReS model
    print("\n== Training TaReS model ==")
    model_tares, hist_tares = train_model(train_loader, val_loader, n_classes=2, device=device, epochs=epochs, lr=lr,
                                          tares=True, BT=BT, sigma_T=sigma_T, sigma_N=sigma_N, lambda_R=lambda_R)
    plot_training_curves(hist_tares, condition='tares', outdir=images_dir)
    tares_path = os.path.join(models_dir, 'simplecnn_tares.pt')
    torch.save(model_tares.state_dict(), tares_path)

    # Save training histories
    with open(os.path.join(models_dir, 'training_histories.json'), 'w') as f:
        json.dump({'baseline': hist_base, 'tares': hist_tares}, f, indent=2)

    print(f"Saved baseline model to {base_path} and TaReS model to {tares_path}.")
    print("Training complete.")


if __name__ == '__main__':
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config/config.yaml'
    run(cfg_path)
