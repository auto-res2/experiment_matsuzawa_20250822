import os
import argparse
from typing import Tuple

import torch
import yaml
import numpy as np


def ensure_dirs():
    os.makedirs('data', exist_ok=True)
    os.makedirs('.research/iteration1/images', exist_ok=True)


def set_seed(seed: int = 1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


def generate_sbm(N: int = 1000, d: int = 64, K: int = 5, p_in: float = 0.08, p_out: float = 0.01, feat_snr: float = 2.0, seed: int = 1):
    set_seed(seed)
    n_per = [N // K] * K
    for i in range(N % K):
        n_per[i] += 1
    y = torch.empty(N, dtype=torch.long)
    start = 0
    for c, n in enumerate(n_per):
        y[start:start+n] = c
        start += n
    perm = torch.randperm(N)
    y = y[perm]

    idx_by_class = [(y == c).nonzero(as_tuple=False).view(-1) for c in range(K)]
    edges = []
    for a in range(K):
        Ia = idx_by_class[a]
        for b in range(a, K):
            Ib = idx_by_class[b]
            pa = p_in if a == b else p_out
            if Ia.numel() == 0 or Ib.numel() == 0:
                continue
            if a == b:
                U = torch.rand(Ia.numel(), Ia.numel()) < pa
                U = torch.triu(U, diagonal=1)
                ii, jj = torch.nonzero(U, as_tuple=True)
                ea = Ia[ii]; eb = Ia[jj]
            else:
                U = torch.rand(Ia.numel(), Ib.numel()) < pa
                ii, jj = torch.nonzero(U, as_tuple=True)
                ea = Ia[ii]; eb = Ib[jj]
            if ea.numel() > 0:
                edges.append(torch.stack([ea, eb], dim=0))
    if len(edges) > 0:
        E_und = torch.cat(edges, dim=1)
        E_rev = torch.stack([E_und[1], E_und[0]], dim=0)
        ei = torch.cat([E_und, E_rev], dim=1)
    else:
        ei = torch.empty(2, 0, dtype=torch.long)
    mask = ei[0] != ei[1]
    ei = ei[:, mask]

    means = torch.randn(K, d) * feat_snr
    X = means[y] + torch.randn(N, d)
    X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)
    return X, y, ei


def train_val_test_split(N: int, train_ratio=0.6, val_ratio=0.2, seed: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    set_seed(seed)
    idx = torch.randperm(N)
    n_tr = int(train_ratio * N)
    n_val = int(val_ratio * N)
    idx_train = idx[:n_tr]
    idx_val = idx[n_tr:n_tr+n_val]
    idx_test = idx[n_tr+n_val:]
    return idx_train, idx_val, idx_test


def preprocess_dataset(config_path: str):
    ensure_dirs()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ds = config['dataset']
    N = int(ds['N']); d = int(ds['d']); K = int(ds['K'])
    p_in = float(ds['p_in']); p_out = float(ds['p_out']); feat_snr = float(ds['feat_snr'])
    seed = int(config['seed'])

    print('[Preprocess] Generating SBM synthetic dataset...')
    X, y, edge_index = generate_sbm(N=N, d=d, K=K, p_in=p_in, p_out=p_out, feat_snr=feat_snr, seed=seed)
    idx_train, idx_val, idx_test = train_val_test_split(N, train_ratio=float(ds['train_ratio']), val_ratio=float(ds['val_ratio']), seed=seed)

    data = {
        'X': X,
        'y': y,
        'edge_index': edge_index,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
    }
    out_path = config['data']['path']
    torch.save(data, out_path)
    print(f"[Preprocess] Saved dataset -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data (synthetic SBM)')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to YAML config')
    args = parser.parse_args()
    preprocess_dataset(args.config)
