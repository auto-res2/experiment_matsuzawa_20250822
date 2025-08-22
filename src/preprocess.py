import os
import sys
import json
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# High-quality PDF outputs
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


class SyntheticManifoldImages(Dataset):
    def __init__(self, n: int = 2000, pattern: str = 'blobs', normal_noise: float = 0.05, seed: int = 0):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.n = n
        self.pattern = pattern
        self.normal_noise = normal_noise
        self.size = 16
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


def compute_BT_pca(dataloader: DataLoader, k: int = 4, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    X_list = []
    for xb, _ in dataloader:
        X_list.append(xb.view(xb.size(0), -1).to(device))
    X = torch.cat(X_list, dim=0)
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh.T  # [d, d]
    BT = V[:, :k]
    BT, _ = torch.linalg.qr(BT)
    # Explained variance per component approximation
    var_explained = (S[:k] ** 2) / (S ** 2).sum()
    return BT, var_explained.detach().cpu()


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

    # Geometry config
    k = int(cfg['geometry'].get('k', 4))

    # Build dataset for PCA
    train_ds = SyntheticManifoldImages(n=n_train, pattern=pattern, normal_noise=normal_noise, seed=0)
    tmp_loader_for_pca = DataLoader(train_ds, batch_size=128, shuffle=False)

    BT, var_explained = compute_BT_pca(tmp_loader_for_pca, k=k, device=device)

    # Save BT
    bt_path = os.path.join(data_dir, f'BT_k{k}.pt')
    torch.save(BT.to('cpu'), bt_path)

    # Save diagnostic plot
    plt.figure()
    xs = np.arange(1, k + 1)
    plt.bar(xs, var_explained.numpy())
    plt.xlabel('PC index (tangent axis)')
    plt.ylabel('Explained variance ratio')
    plt.title('Tangent axes via PCA')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'pca_tangent_axes_explained_variance.pdf'))
    plt.close()

    # Write meta info
    meta = {
        'pattern': pattern,
        'normal_noise': normal_noise,
        'k': k,
        'var_explained': var_explained.numpy().tolist()
    }
    with open(os.path.join(data_dir, f'BT_k{k}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved BT (k={k}) to {bt_path} and PCA diagnostics to images.")


if __name__ == '__main__':
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config/config.yaml'
    run(cfg_path)
