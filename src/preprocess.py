# Preprocessing and data utilities (src/preprocess.py)
# - Synthetic dataset for quick tests
# - Non-IID and IID partitioning helpers
# - Seed control and directory setup

import os
import random
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset

try:
    from torchvision import datasets, transforms
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SyntheticImageDataset(Dataset):
    """Small synthetic dataset for quick tests.
    Creates num_classes classes of tiny RGB images with simple patterns.
    """
    def __init__(self, n_per_class=50, img_size=8, num_classes=10, seed=0):
        rng = np.random.default_rng(seed)
        self.num_classes = num_classes
        self.images = []
        self.labels = []
        for c in range(num_classes):
            for _ in range(n_per_class):
                base = rng.random((3, img_size, img_size), dtype=np.float32)
                base += (c / max(num_classes-1, 1)) * 0.25
                base = np.clip(base, 0, 1)
                self.images.append(base.astype(np.float32))
                self.labels.append(c)
        self.images = np.stack(self.images)
        self.labels = np.array(self.labels, dtype=np.int64)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        import torch
        x = torch.tensor(self.images[idx])
        y = int(self.labels[idx])
        return x, y


def dirichlet_noniid_split(labels: np.ndarray, num_clients: int, alpha: float = 0.1, min_size: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = int(labels.max()) + 1
    idx_by_class = [np.where(labels == k)[0] for k in range(K)]
    for idx in idx_by_class:
        rng.shuffle(idx)
    proportions = rng.dirichlet(alpha=np.ones(num_clients), size=K)  # K x C
    client_indices = [[] for _ in range(num_clients)]
    for k in range(K):
        idx = idx_by_class[k]
        sizes = np.floor(proportions[k] / proportions[k].sum() * len(idx)).astype(int)
        while sizes.sum() < len(idx):
            sizes[rng.integers(0, num_clients)] += 1
        while sizes.sum() > len(idx):
            j = rng.integers(0, num_clients)
            if sizes[j] > 0:
                sizes[j] -= 1
        start = 0
        for c in range(num_clients):
            end = start + sizes[c]
            client_indices[c].extend(idx[start:end].tolist())
            start = end
    for c in range(num_clients):
        if len(client_indices[c]) < min_size:
            needed = min_size - len(client_indices[c])
            pool = np.setdiff1d(np.arange(len(labels)), np.array(client_indices[c], dtype=int), assume_unique=False)
            if len(pool) > 0:
                add = rng.choice(pool, size=min(needed, len(pool)), replace=False)
                client_indices[c].extend(add.tolist())
    return [np.array(sorted(list(set(idx))), dtype=int) for idx in client_indices]


def iid_split(num_samples: int, num_clients: int, min_size: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    res = []
    for arr in splits:
        if len(arr) < min_size:
            extra = rng.choice(indices, size=min_size - len(arr), replace=False)
            arr = np.concatenate([arr, extra])
        res.append(np.array(sorted(list(set(arr))), dtype=int))
    return res


def ensure_dirs():
    os.makedirs('.research/iteration1/images', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
