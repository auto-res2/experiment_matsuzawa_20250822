import os
import sys
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Ensure high-quality PDF outputs
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
# Synthetic dataset
# ------------------------------

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


def proj_N(v: torch.Tensor, BT: torch.Tensor) -> torch.Tensor:
    return v - proj_T(v, BT)


def whiten_x(x_flat: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float) -> torch.Tensor:
    PT_x = proj_T(x_flat, BT)
    PN_x = x_flat - PT_x
    return PT_x / sigma_T + PN_x / sigma_N


def dewhiten_delta_y(delta_y: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float) -> torch.Tensor:
    PT_y = proj_T(delta_y, BT)
    PN_y = delta_y - PT_y
    return PT_y * sigma_T + PN_y * sigma_N


# ------------------------------
# Lipschitz upper bound
# ------------------------------

@torch.no_grad()
def spectral_norm_upper_bound(model: nn.Module, power_steps: int = 20) -> float:
    bounds = []
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            W = mod.weight.data
            W_mat = W.reshape(W.size(0), -1)
            u = torch.randn(W_mat.size(0), device=W_mat.device)
            u = u / (u.norm() + 1e-12)
            for _ in range(power_steps):
                v = W_mat.t().matmul(u)
                v = v / (v.norm() + 1e-12)
                u = W_mat.matmul(v)
                u = u / (u.norm() + 1e-12)
            sigma = torch.dot(u, W_mat.matmul(v)).abs().item()
            bounds.append(max(sigma, 1e-6))
    Lglob = 1.0
    for b in bounds:
        Lglob *= b
    return float(Lglob)


# ------------------------------
# Smoothed classifier and certification (y-space)
# ------------------------------

def smoothed_logits_y(model: nn.Module, x0_img: torch.Tensor, y_delta: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float, M: int = 512) -> torch.Tensor:
    B = x0_img.size(0)
    device = x0_img.device
    d = x0_img.numel() // B
    eps = torch.randn(M // 2, B, d, device=device)
    eps = torch.cat([eps, -eps], dim=0)
    y_full = y_delta.unsqueeze(0) + eps  # [M, B, d]
    delta_x = dewhiten_delta_y(y_full.reshape(-1, d), BT, sigma_T, sigma_N)
    x_noisy = x0_img.view(B, -1).unsqueeze(0).repeat(M, 1, 1).reshape(-1, d) + delta_x
    x_noisy_img = x_noisy.view(M, B, *x0_img.shape[1:])
    logits_chunks = []
    for m in torch.split(x_noisy_img, 64, dim=0):
        Bm = m.size(0)
        m_flat = m.reshape(Bm * B, *x0_img.shape[1:])
        logits_m = model(m_flat).view(Bm, B, -1)
        logits_chunks.append(logits_m)
    logits = torch.cat(logits_chunks, dim=0).mean(dim=0)
    return logits


def estimate_gap_and_proj_grad_y(model: nn.Module, x0_img: torch.Tensor, true_y: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float, M: int = 512) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = x0_img.size(0)
    d = x0_img.numel() // B
    y_delta = torch.zeros(B, d, device=x0_img.device, requires_grad=True)
    logits = smoothed_logits_y(model, x0_img, y_delta, BT, sigma_T, sigma_N, M=M)
    top2 = torch.topk(logits, k=2, dim=1).indices
    c0 = top2[:, 0]
    c1 = torch.where(c0 == true_y, top2[:, 1], c0)
    gap = logits[torch.arange(B), c0] - logits[torch.arange(B), c1]
    true_logit = logits[torch.arange(B), true_y].sum()
    grad = torch.autograd.grad(true_logit, y_delta, retain_graph=False, create_graph=False)[0]
    gT = proj_T(grad, BT)
    gN = grad - gT
    normT = gT.norm(dim=1)
    normN = gN.norm(dim=1)
    return gap.detach(), normT.detach(), normN.detach()


def hocert_radius_y(gap_lo: torch.Tensor, normT_up: torch.Tensor, normN_up: torch.Tensor) -> torch.Tensor:
    denom = torch.sqrt(normT_up ** 2 + normN_up ** 2 + 1e-12)
    ry = gap_lo / (denom + 1e-12)
    ry = torch.clamp(ry, min=0.0)
    return ry


def certify_batch(model: nn.Module, x_batch: torch.Tensor, y_batch: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float, Lglob: float = None, M: int = 512, ci_factor: float = 1.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.enable_grad():
        gap, normT, normN = estimate_gap_and_proj_grad_y(model, x_batch, y_batch, BT, sigma_T, sigma_N, M=M)
        if Lglob is not None:
            normN = torch.minimum(normN, torch.full_like(normN, float(Lglob)))
        gap_lo = gap / ci_factor
        normT_up = normT * ci_factor
        normN_up = normN * ci_factor
        ry = hocert_radius_y(gap_lo, normT_up, normN_up)
        rL2 = ry * min(sigma_T, sigma_N)
    return ry.detach(), rL2.detach(), normT.detach(), normN.detach()


# ------------------------------
# Evaluation utilities
# ------------------------------

def plot_hist(data: torch.Tensor, title: str, fname: str, outdir: str):
    ensure_dir(outdir)
    plt.figure()
    sns.histplot(data.cpu().numpy(), kde=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{fname}.pdf'))
    plt.close()


def plot_scatter(x: torch.Tensor, y: torch.Tensor, xlabel: str, ylabel: str, fname: str, outdir: str):
    ensure_dir(outdir)
    plt.figure()
    plt.scatter(x.cpu().numpy(), y.cpu().numpy(), s=10, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{fname}.pdf'))
    plt.close()


def plot_confusion_matrix(model: nn.Module, loader: DataLoader, n_classes: int, fname: str, outdir: str):
    ensure_dir(outdir)
    cm = torch.zeros(n_classes, n_classes)
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(next(model.parameters()).device)).argmax(dim=1).cpu()
            for t, p in zip(yb, preds):
                cm[t.long(), p.long()] += 1
    cm = cm.numpy()
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{fname}.pdf'))
    plt.close()


def evaluate_certificates(model: nn.Module, loader: DataLoader, BT: torch.Tensor, sigma_T: float, sigma_N: float, Lglob: float = None, M: int = 512, ry_list: List[float] = [0.5, 1.0, 1.5]) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_ry, all_rL2, all_S, all_labels, all_preds = [], [], [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(BT.device), yb.to(BT.device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        mask = preds.eq(yb)
        if mask.sum() == 0:
            continue
        ry, rL2, nT, nN = certify_batch(model, xb[mask], yb[mask], BT, sigma_T, sigma_N, Lglob=Lglob, M=M)
        all_ry.append(ry.cpu())
        all_rL2.append(rL2.cpu())
        S = nN / (nT + 1e-12)
        all_S.append(S.cpu())
        all_labels.append(yb[mask].cpu())
        all_preds.append(preds[mask].cpu())
    if len(all_ry) == 0:
        print('Warning: no correctly classified samples to certify.')
        return {}, torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    all_ry = torch.cat(all_ry)
    all_rL2 = torch.cat(all_rL2)
    all_S = torch.cat(all_S)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    metrics = {}
    for r in ry_list:
        metrics[f'cert_acc_ry_{r:.2f}'] = float((all_ry >= r).float().mean())
    L2_list = [r * min(sigma_T, sigma_N) for r in ry_list]
    for r in L2_list:
        metrics[f'cert_acc_L2_{r:.2f}'] = float((all_rL2 >= r).float().mean())
    metrics['ry_mean'] = float(all_ry.mean())
    metrics['rL2_mean'] = float(all_rL2.mean())
    metrics['ry_p50'] = float(all_ry.quantile(0.5))
    metrics['ry_p90'] = float(all_ry.quantile(0.9))
    metrics['S_mean'] = float(all_S.mean())
    metrics['S_p90'] = float(all_S.quantile(0.9))
    return metrics, all_ry, all_rL2, all_S, all_labels


# ------------------------------
# PN-focused attack (y-space PGD)
# ------------------------------

def project_to_l2_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
    B = delta.size(0)
    flat = delta.view(B, -1)
    norms = flat.norm(dim=1, keepdim=True) + 1e-12
    scale = torch.clamp(eps / norms, max=1.0)
    return (flat * scale).view_as(delta)


@torch.enable_grad()
def smoothed_margin_grad_y(model: nn.Module, x0_img: torch.Tensor, y_delta: torch.Tensor, true_label: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float, M: int = 256) -> torch.Tensor:
    logits = smoothed_logits_y(model, x0_img, y_delta, BT, sigma_T, sigma_N, M=M)
    top2 = torch.topk(logits, k=2, dim=1).indices
    c0 = top2[:, 0]
    c1 = torch.where(c0 == true_label, top2[:, 1], c0)
    margin = logits[torch.arange(y_delta.size(0)), c1] - logits[torch.arange(y_delta.size(0)), true_label]
    grad = torch.autograd.grad(margin.mean(), y_delta, retain_graph=False, create_graph=False)[0]
    return grad


def pgd_pn_y(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float, eps_y: float, steps: int = 30, restarts: int = 2, step_frac: float = 0.25, mixed: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = xb.device
    B = xb.size(0)
    d = xb.numel() // B
    best_delta = torch.zeros(B, d, device=device)
    best_success = torch.zeros(B, dtype=torch.bool, device=device)
    best_pred = yb.clone()
    for r in range(restarts):
        y_delta = torch.randn(B, d, device=device)
        y_delta = project_to_l2_ball(y_delta, eps_y)
        y_delta.requires_grad_(True)
        alpha = eps_y * step_frac
        for _ in range(steps):
            grad = smoothed_margin_grad_y(model, xb, y_delta, yb, BT, sigma_T, sigma_N, M=256)
            gT = proj_T(grad, BT)
            gN = grad - gT
            if mixed:
                g = 0.8 * gN + 0.2 * gT
            else:
                g = gN
            g = g / (g.norm(dim=1, keepdim=True) + 1e-12)
            with torch.no_grad():
                y_delta = y_delta + alpha * g
                y_delta = project_to_l2_ball(y_delta, eps_y)
                y_delta.requires_grad_(True)
        with torch.no_grad():
            delta_x = dewhiten_delta_y(y_delta, BT, sigma_T, sigma_N)
            x_adv = xb.view(B, -1) + delta_x
            logits = model(x_adv.view_as(xb))
            pred = logits.argmax(dim=1)
            succ = ~pred.eq(yb)
            replace = succ & ~best_success
            best_delta[replace] = y_delta[replace]
            best_success |= succ
            best_pred[replace] = pred[replace]
    PN_energy = torch.zeros(B, device=device)
    if best_success.any():
        gT = proj_T(best_delta, BT)
        gN = best_delta - gT
        PN_energy = (gN.norm(dim=1) / (best_delta.norm(dim=1) + 1e-12)).detach()
    return best_delta.detach(), best_success.detach(), PN_energy.detach()


# ------------------------------
# Transitivity along tangent axes
# ------------------------------

@torch.no_grad()
def mc_true_confidence(model: nn.Module, xb: torch.Tensor, y_delta: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float, M: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = smoothed_logits_y(model, xb, y_delta, BT, sigma_T, sigma_N, M=M)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return conf, pred


@torch.no_grad()
def find_axis_point(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, a_axis: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float, t_max: float = 2.5, M: int = 512, tau: float = 0.05, iters: int = 10) -> torch.Tensor:
    B = xb.size(0)
    d = xb.numel() // B
    t_lo = torch.zeros(B, device=xb.device)
    t_hi = torch.full((B,), t_max, device=xb.device)
    for _ in range(iters):
        t_mid = 0.5 * (t_lo + t_hi)
        y_delta = (t_mid.unsqueeze(1) * a_axis).detach()
        conf, pred = mc_true_confidence(model, xb, y_delta, BT, sigma_T, sigma_N, M=M)
        ok = (pred == yb) & (conf > tau)
        t_lo = torch.where(ok, t_mid, t_lo)
        t_hi = torch.where(ok, t_hi, t_mid)
    return (t_lo.unsqueeze(1) * a_axis)


@torch.no_grad()
def transitivity_union_cert(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, BT: torch.Tensor, sigma_T: float, sigma_N: float, m_axes: int = 2, M_cert: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    B = xb.size(0)
    d = xb.numel() // B
    axes = BT[:, :m_axes].t().contiguous()
    centers = [torch.zeros(B, d, device=xb.device)]
    for i in range(m_axes):
        a = axes[i].unsqueeze(0).repeat(B, 1)
        y_axis = find_axis_point(model, xb, yb, a, BT, sigma_T, sigma_N, t_max=2.5, M=512, tau=0.05, iters=10)
        centers.append(y_axis)
    ry_list, rL2_list = [], []
    for y_center in centers:
        delta_x = dewhiten_delta_y(y_center, BT, sigma_T, sigma_N)
        x_center = xb.view(B, -1) + delta_x
        ry, rL2, _, _ = certify_batch(model, x_center.view_as(xb), yb, BT, sigma_T, sigma_N, Lglob=None, M=M_cert)
        ry_list.append(ry)
        rL2_list.append(rL2)
    ry_stack = torch.stack(ry_list, dim=0)
    rL2_stack = torch.stack(rL2_list, dim=0)
    ry_union = ry_stack.max(dim=0).values
    rL2_union = rL2_stack.max(dim=0).values
    return ry_union.cpu(), rL2_union.cpu()


# ------------------------------
# Main evaluation runner
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

    # Data config
    pattern = cfg['data'].get('pattern', 'blobs')
    normal_noise = float(cfg['data'].get('normal_noise', 0.05))
    n_test = int(cfg['data'].get('n_test', 300))

    # Geometry config
    k = int(cfg['geometry'].get('k', 4))
    sigma_T = float(cfg['geometry'].get('sigma_T', 0.25))
    sigma_N = float(cfg['geometry'].get('sigma_N', 0.15))

    # Load BT
    bt_path = os.path.join(data_dir, f'BT_k{k}.pt')
    if not os.path.exists(bt_path):
        raise FileNotFoundError(f"BT file not found at {bt_path}. Run preprocess first.")
    BT = torch.load(bt_path, map_location=device)

    # Load models
    base_path = os.path.join(models_dir, 'simplecnn_baseline.pt')
    tares_path = os.path.join(models_dir, 'simplecnn_tares.pt')
    if not os.path.exists(base_path) or not os.path.exists(tares_path):
        raise FileNotFoundError('Trained model weights not found. Run train first.')

    model_base = SimpleCNN(n_classes=2).to(device)
    model_base.load_state_dict(torch.load(base_path, map_location=device))
    model_base.eval()

    model_tares = SimpleCNN(n_classes=2).to(device)
    model_tares.load_state_dict(torch.load(tares_path, map_location=device))
    model_tares.eval()

    # Datasets
    test_ds = SyntheticManifoldImages(n=n_test, pattern=pattern, normal_noise=normal_noise, seed=20)
    test_loader = DataLoader(test_ds, batch_size=int(cfg['eval'].get('batch_size', 128)), shuffle=False)

    # MC settings
    M_cert = int(cfg['eval'].get('M_cert', 256))
    ry_list = list(map(float, cfg['eval'].get('ry_thresholds', [0.25, 0.5, 0.75])))

    # Lipschitz bounds
    Lglob_base = spectral_norm_upper_bound(model_base)
    Lglob_tares = spectral_norm_upper_bound(model_tares)
    print(f"[Eval] Lglob baseline={Lglob_base:.4e} tares={Lglob_tares:.4e}")

    # Experiment 1: core certification
    metrics_base, ry_b, rL2_b, S_b, _ = evaluate_certificates(model_base, test_loader, BT, sigma_T, sigma_N, Lglob=Lglob_base, M=M_cert, ry_list=ry_list)
    metrics_tares, ry_t, rL2_t, S_t, _ = evaluate_certificates(model_tares, test_loader, BT, sigma_T, sigma_N, Lglob=Lglob_tares, M=M_cert, ry_list=ry_list)

    print('\n[Exp1] Baseline metrics:')
    for k, v in metrics_base.items():
        print(f"  {k}: {v:.4f}")
    print('[Exp1] TaReS metrics:')
    for k, v in metrics_tares.items():
        print(f"  {k}: {v:.4f}")

    # Save plots for exp1
    plot_confusion_matrix(model_base, test_loader, n_classes=2, fname='confusion_matrix_baseline', outdir=images_dir)
    plot_confusion_matrix(model_tares, test_loader, n_classes=2, fname='confusion_matrix_tares', outdir=images_dir)
    plot_hist(ry_b, 'ry distribution (baseline)', 'ry_baseline', images_dir)
    plot_hist(rL2_b, 'rL2 distribution (baseline)', 'rL2_baseline', images_dir)
    plot_hist(S_b, 'S-index distribution (baseline)', 'sindex_baseline', images_dir)
    plot_hist(ry_t, 'ry distribution (tares)', 'ry_tares', images_dir)
    plot_hist(rL2_t, 'rL2 distribution (tares)', 'rL2_tares', images_dir)
    plot_hist(S_t, 'S-index distribution (tares)', 'sindex_tares', images_dir)
    plot_scatter(ry_b, rL2_b, 'ry', 'rL2', 'ry_vs_rL2_baseline', images_dir)
    plot_scatter(ry_t, rL2_t, 'ry', 'rL2', 'ry_vs_rL2_tares', images_dir)

    results = {
        'exp1': {
            'baseline': metrics_base,
            'tares': metrics_tares,
            'sigma_T': sigma_T,
            'sigma_N': sigma_N,
            'pattern': pattern
        }
    }

    # Experiment 2: Spoof-resistance (PN attacks)
    # Prepare subset of correctly classified for baseline model
    xb_list, yb_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model_base(xb.to(device)).argmax(dim=1).cpu()
            mask = pred.eq(yb)
            if mask.sum() > 0:
                xb_list.append(xb[mask])
                yb_list.append(yb[mask])
    if xb_list:
        Xc = torch.cat(xb_list, dim=0).to(device)
        Yc = torch.cat(yb_list, dim=0).to(device)
        n_eval = min(int(cfg['eval'].get('attack_n_eval', 64)), Xc.size(0))
        Xc = Xc[:n_eval]
        Yc = Yc[:n_eval]
        eps_y_list = list(map(float, cfg['eval'].get('attack_eps_y_list', [0.25, 0.5, 0.75])))
        results['exp2'] = {}
        for name, model in [('baseline', model_base), ('tares', model_tares)]:
            asr_pn, asr_mixed, calib_break_rates = [], [], []
            Lglob = spectral_norm_upper_bound(model)
            ry, rL2, _, _ = certify_batch(model, Xc, Yc, BT, sigma_T, sigma_N, Lglob=Lglob, M=M_cert)
            for eps in eps_y_list:
                _, succ_pn, _ = pgd_pn_y(model, Xc, Yc, BT, sigma_T, sigma_N, eps_y=eps, steps=int(cfg['eval'].get('attack_steps', 20)), restarts=2, step_frac=0.25, mixed=False)
                asr_pn.append(float(succ_pn.float().mean().cpu()))
                _, succ_mx, _ = pgd_pn_y(model, Xc, Yc, BT, sigma_T, sigma_N, eps_y=eps, steps=int(cfg['eval'].get('attack_steps', 20)), restarts=2, step_frac=0.25, mixed=True)
                asr_mixed.append(float(succ_mx.float().mean().cpu()))
                cert_mask = ry.cpu() >= eps
                broken = succ_pn.cpu() & cert_mask
                calib_break_rates.append(float(broken.float().mean()))
                print(f"[Exp2][{name}] eps_y={eps:.2f} ASR_PN={asr_pn[-1]:.3f} ASR_Mixed={asr_mixed[-1]:.3f} CalibBreak={calib_break_rates[-1]:.3f}")
            # Plot curves
            plt.figure()
            plt.plot(eps_y_list, asr_pn, marker='o', label='PN-PGD')
            plt.plot(eps_y_list, asr_mixed, marker='s', label='Mixed-PGD')
            plt.xlabel('epsilon_y')
            plt.ylabel('Attack Success Rate')
            plt.legend()
            plt.title(f'ASR curves ({name})')
            plt.tight_layout()
            plt.savefig(os.path.join(images_dir, f'attack_success_vs_epsilon_{name}.pdf'))
            plt.close()

            plt.figure()
            plt.plot(eps_y_list, calib_break_rates, marker='^', color='red')
            plt.xlabel('epsilon_y')
            plt.ylabel('Certified-yet-broken rate')
            plt.title(f'Calibration proxy ({name})')
            plt.tight_layout()
            plt.savefig(os.path.join(images_dir, f'calibration_curve_{name}.pdf'))
            plt.close()

            results['exp2'][name] = {
                'ASR_PN': asr_pn,
                'ASR_Mixed': asr_mixed,
                'CalibBreak': calib_break_rates
            }
    else:
        print('[Exp2] No correctly classified points found for baseline; skipping attacks.')

    # Experiment 3: Transitivity unions
    # Correctly classified subset again (baseline)
    X_list, Y_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model_base(xb.to(device)).argmax(dim=1).cpu()
            mask = pred.eq(yb)
            if mask.sum() > 0:
                X_list.append(xb[mask])
                Y_list.append(yb[mask])
    if X_list:
        Xc = torch.cat(X_list, dim=0).to(device)
        Yc = torch.cat(Y_list, dim=0).to(device)
        n_eval = min(int(cfg['eval'].get('trans_n_eval', 64)), Xc.size(0))
        Xc = Xc[:n_eval]
        Yc = Yc[:n_eval]
        results['exp3'] = {}
        for name, model in [('baseline', model_base), ('tares', model_tares)]:
            ry_union2, rL2_union2 = transitivity_union_cert(model, Xc, Yc, BT, sigma_T, sigma_N, m_axes=2, M_cert=M_cert)
            ry_union4, rL2_union4 = transitivity_union_cert(model, Xc, Yc, BT, sigma_T, sigma_N, m_axes=4, M_cert=M_cert)
            ry_base, rL2_base, _, _ = certify_batch(model, Xc, Yc, BT, sigma_T, sigma_N, Lglob=None, M=M_cert)
            ca_base = float((ry_base >= 0.5).float().mean())
            ca_u2 = float((ry_union2 >= 0.5).float().mean())
            ca_u4 = float((ry_union4 >= 0.5).float().mean())
            print(f"[Exp3][{name}] Certified acc @ ry=0.5: base={ca_base:.3f} union2={ca_u2:.3f} union4={ca_u4:.3f}")
            plot_hist(ry_union2, f'ry union (m=2) {name}', f'ry_union_m2_{name}', images_dir)
            plot_hist(ry_union4, f'ry union (m=4) {name}', f'ry_union_m4_{name}', images_dir)
            plot_hist(rL2_union2, f'rL2 union (m=2) {name}', f'rL2_union_m2_{name}', images_dir)
            plot_hist(rL2_union4, f'rL2 union (m=4) {name}', f'rL2_union_m4_{name}', images_dir)
            results['exp3'][name] = {
                'cert_acc_ry0.5_base': ca_base,
                'cert_acc_ry0.5_union2': ca_u2,
                'cert_acc_ry0.5_union4': ca_u4,
                'ry_base_mean': float(ry_base.mean().cpu()),
                'ry_union2_mean': float(ry_union2.mean().cpu()),
                'ry_union4_mean': float(ry_union4.mean().cpu())
            }
    else:
        print('[Exp3] No correctly classified samples to certify; skipping transitivity.')

    # Save JSON results
    ensure_dir('.research/iteration1')
    results_path = os.path.join('.research/iteration1', 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {results_path}")


if __name__ == '__main__':
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config/config.yaml'
    run(cfg_path)
