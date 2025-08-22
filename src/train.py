import os
import json
import time
import math
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLPRegressor(nn.Module):
    def __init__(self, d_in: int, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        last = d_in
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Directories
    data_dir = cfg.get("data", {}).get("dir", "data")
    models_dir = cfg.get("model", {}).get("dir", "models")
    images_dir = cfg.get("experiment", {}).get("images_dir", ".research/iteration1/images")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    # Load dataset
    data_name = cfg.get("data", {}).get("name", "synthetic_tabular")
    data_path = os.path.join(data_dir, f"{data_name}.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}. Run preprocess.py first.")
    npz = np.load(data_path)
    X_train = npz["X_train"].astype(np.float32)
    y_train = npz["y_train"].astype(np.float32)
    X_val = npz["X_val"].astype(np.float32)
    y_val = npz["y_val"].astype(np.float32)

    # Normalize features by train stats
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std

    d_in = X_train.shape[1]

    hidden_dims = tuple(cfg.get("model", {}).get("hidden_dims", [128, 64]))
    lr = float(cfg.get("model", {}).get("lr", 1e-3))
    epochs = int(cfg.get("model", {}).get("epochs", 5))
    batch_size = int(cfg.get("model", {}).get("batch_size", 128))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(d_in, hidden_dims).to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_n), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                vloss += loss.item() * xb.size(0)
            vloss /= len(val_loader.dataset)
            val_losses.append(vloss)

        print(f"[Train] Epoch {epoch:02d}/{epochs}  Train MSE={epoch_loss:.5f}  Val MSE={vloss:.5f}")

    t1 = time.time()
    print(f"[Train] Done in {(t1 - t0):.2f}s")

    # Save model
    model_path = os.path.join(models_dir, "mlp_regressor.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "d_in": d_in,
        "hidden_dims": hidden_dims,
        "x_mean": x_mean,
        "x_std": x_std,
        "seed": seed,
    }, model_path)
    print(f"[Train] Saved model to {model_path}")

    # Save training curve
    plt.figure(figsize=(5, 4))
    plt.plot(train_losses, label="Train MSE", marker="o", ms=3)
    plt.plot(val_losses, label="Val MSE", marker="s", ms=3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Curve (MLP Regressor)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(images_dir, "training_curve_mlp.pdf")
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    print(f"[Train] Saved {curve_path}")


if __name__ == "__main__":
    train_model()
