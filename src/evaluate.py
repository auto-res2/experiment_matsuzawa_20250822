import os
import argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from train import MiGAD, build_row_stochastic, precompute_multi_hop, build_budget


def ensure_dirs():
    os.makedirs('.research/iteration1/images', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)


def plot_heatmap(matrix: np.ndarray, title: str, fname: str, cmap: str = 'viridis'):
    plt.figure()
    plt.imshow(matrix, cmap=cmap, aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Col')
    plt.ylabel('Row')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'[Figure saved] {fname}')


def evaluate_migad(config: Dict):
    ensure_dirs()
    device_str = config['train'].get('device', 'auto')
    if device_str == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_str
    print(f"[Device] Using {device}")

    # Load data
    data = torch.load(config['data']['path'])
    X = data['X'].float().to(device)
    y = data['y'].long().to(device)
    edge_index = data['edge_index'].long()
    idx_train = data['idx_train'].long()
    idx_val = data['idx_val'].long()
    idx_test = data['idx_test'].long()

    N, d_in = X.size()
    num_classes = int(y.max().item()) + 1

    # P
    P = build_row_stochastic(edge_index, N).to(device)

    # Load model
    ckpt_path = os.path.join('models', config['output']['model_name'])
    ckpt = torch.load(ckpt_path, map_location=device)
    L = int(config['model']['L'])
    G = int(config['model']['G'])
    hidden = int(config['model']['hidden'])
    top_rho = int(config['model']['top_rho'])

    model = MiGAD(d_in, hidden, num_classes, G, L, top_rho=top_rho, rank=int(config['model'].get('rank', 8)), risk_mode='align').to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Build budgets
    with torch.no_grad():
        S_now = model.group_head(X)
        rho = build_budget(P, S_now, rho0=float(config['train']['rho0']), rho_min=float(config['train']['rho_min']), rho_max=float(config['train']['rho_max']))
        logits, aux = model(P, X, rho, temp_d=0.6, hard_kappa=float(config['train']['hard_kappa_cool']))

    # Accuracies
    def acc_split(split_idx):
        pred = logits.argmax(dim=1)
        return (pred[split_idx] == y[split_idx]).float().mean().item()

    acc_train = acc_split(idx_train)
    acc_val = acc_split(idx_val)
    acc_test = acc_split(idx_test)
    print(f"[Eval] Accuracy (train/val/test): {acc_train:.3f}/{acc_val:.3f}/{acc_test:.3f}")

    # Confusion matrix on test
    pred = logits.argmax(dim=1)
    K = num_classes
    cm = np.zeros((K, K), dtype=np.int64)
    for i in idx_test.tolist():
        cm[int(y[i].item()), int(pred[i].item())] += 1
    img_dir = '.research/iteration1/images'
    plot_heatmap(cm, 'Confusion Matrix (MiGAD, test)', os.path.join(img_dir, 'confusion_matrix_migad.pdf'), cmap='Blues')

    # Hop weight average
    w = aux['w'].detach().cpu().numpy()
    plt.figure(); plt.plot(np.arange(w.shape[1]), w.mean(axis=0), marker='o'); plt.xlabel('hop k'); plt.ylabel('avg weight'); plt.title('MiGAD Avg Hop Weights'); plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(img_dir, 'hop_weights_migad.pdf'), bbox_inches='tight'); plt.close(); print('[Figure saved] hop_weights_migad.pdf')

    # Compatibility class-level heatmap
    C = aux['C']  # [G,G]
    S = aux['S']  # [N,G]
    S_cls = torch.zeros(K, C.size(0), device=S.device)
    for c in range(K):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() > 0:
            S_cls[c] = S[idx].mean(dim=0)
    C_cls = (S_cls @ C @ S_cls.t()).detach().cpu().numpy()
    plot_heatmap(C_cls, 'Class-level Compatibility (MiGAD)', os.path.join(img_dir, 'compatibility_heatmap_classes_migad.pdf'))

    # Budget usage diagnostics
    usage = (aux['w'] * aux['R']).sum(dim=1).detach().cpu().numpy()
    plt.figure(); plt.hist(usage, bins=30, color='C0', alpha=0.8); plt.xlabel('Budget usage sum_k w_k r_k'); plt.ylabel('count'); plt.title('Per-node Budget Usage');
    plt.savefig(os.path.join(img_dir, 'budget_usage_hist_migad.pdf'), bbox_inches='tight'); plt.close(); print('[Figure saved] budget_usage_hist_migad.pdf')

    print('[Eval] Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MiGAD model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to YAML config')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    evaluate_migad(config)
