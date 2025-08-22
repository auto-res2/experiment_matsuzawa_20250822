# PACER-FL — Experimental Training Scripts (src/train.py)
# Implements: auditable DDG-TREE noise, residual DP updates with control variates,
# DP gating, FO-aware CountSketch compression, and unified RDP-based budgeting.
# Saves high-quality PDF figures to .research/iteration1/images.

import os
import json
import math
import time
import random
import copy
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# Optional deps
try:
    from torchvision import datasets, transforms
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

try:
    from dp_accounting import rdp as rdp_mod
    _HAS_DP_ACCOUNTING = True
except Exception:
    _HAS_DP_ACCOUNTING = False

try:
    from diffprivlib.mechanisms import Laplace as DPLaplace
    from diffprivlib.tools import histogram as dp_histogram
    _HAS_DIFFPRIVLIB = True
except Exception:
    _HAS_DIFFPRIVLIB = False

from typing import List, Tuple, Dict, Any

# Local imports
from src.preprocess import SyntheticImageDataset, dirichlet_noniid_split, iid_split, set_seed
from src.evaluate import evaluate_model, read_ledger, auditor_reconstruct_noise


# -----------------------------
# Model definition (input-size agnostic via GlobalAvgPool)
# -----------------------------
class SmallCIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# -----------------------------
# Flatten/unflatten utilities
# -----------------------------

def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def assign_params(model: nn.Module, vec: torch.Tensor):
    offset = 0
    for p in model.parameters():
        num = p.numel()
        p.data.copy_(vec[offset:offset+num].reshape(p.shape))
        offset += num


# -----------------------------
# CountSketch and FO-aware dimension selection
# -----------------------------
class CountSketch:
    def __init__(self, d: int, m: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.h = rng.integers(0, m, size=d)
        self.s = rng.choice([-1, 1], size=d)
        self.m = m
        self.d = d
    def S(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(self.m, dtype=np.float32)
        np.add.at(y, self.h, self.s * x)
        return y
    def unS(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(self.h, minlength=self.m).astype(np.float32)
        inv = np.where(counts > 0, 1.0 / counts, 0.0)
        bucket_vals = y * inv
        return bucket_vals[self.h] * self.s


def choose_m_t(d: int, n_t: int, eps_eff_t: float, M_t: float, kappa: float = 1.2) -> int:
    m_est = int(kappa * min(d, max(8, (n_t * eps_eff_t * max(M_t, 1e-6))**2)))
    return int(max(16, min(d, m_est)))


def bits_per_client_countsketch(m: int, value_bits: int = 32, seed_bits: int = 64, extra_overhead_bits: int = 256) -> int:
    return m * value_bits + seed_bits + extra_overhead_bits


# -----------------------------
# Tree schedule + DDG-TREE noise
# -----------------------------

def tree_nodes_for_round(t: int):
    nodes = []
    k, x = 0, t
    while x > 0:
        if x & 1:
            nodes.append((k, (t >> k) << k))
        k += 1
        x >>= 1
    return nodes


def ddg_int_gaussian(shape, sigma, rng: np.random.Generator):
    return np.rint(rng.normal(0.0, sigma, size=shape)).astype(np.int64)


def client_ddg_tree_shares(round_t: int, sketch_dim: int, seeds: dict, target_var_per_node: dict, L_min: int, cohort_size: int):
    rngs = {k: np.random.default_rng(seeds[k]) for k in seeds}
    shares = np.zeros(sketch_dim, dtype=np.int64)
    for node in tree_nodes_for_round(round_t):
        if node not in target_var_per_node:
            continue
        sigma = math.sqrt(target_var_per_node[node])
        scale = math.sqrt(max(L_min, cohort_size))
        z = ddg_int_gaussian(sketch_dim, sigma/scale, rngs[node])
        shares += z
    return shares


# -----------------------------
# DP mechanisms: gating and private quantile
# -----------------------------

def laplace_noise(scale: float, rng: np.random.Generator = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2*abs(u) + 1e-12)


def dp_gate(score: float, theta: float, sensitivity: float, eps: float, rng: np.random.Generator = None) -> bool:
    if _HAS_DIFFPRIVLIB:
        mech = DPLaplace(epsilon=max(eps, 1e-8), sensitivity=max(sensitivity, 1e-12), random_state=rng)
        noisy_score = mech.randomise(score)
        return noisy_score >= theta
    else:
        b = sensitivity / max(eps, 1e-8)
        noisy_score = score + laplace_noise(b, rng=rng)
        return noisy_score >= theta


def private_quantile_from_histogram(norms: np.ndarray, q: float = 0.9, clip_max: float = 10.0, eps: float = 0.05, bins: int = 50, rng: np.random.Generator = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    norms = np.clip(norms, 0.0, clip_max)
    if _HAS_DIFFPRIVLIB:
        counts, edges = dp_histogram(norms, range=(0.0, clip_max), bins=bins, epsilon=max(eps, 1e-8), random_state=rng)
    else:
        counts, edges = np.histogram(norms, range=(0.0, clip_max), bins=bins)
        b = 1.0 / max(eps, 1e-8)
        noise = np.array([laplace_noise(b, rng=rng) for _ in range(len(counts))])
        counts = counts.astype(np.float64) + noise
        counts = np.clip(counts, 0.0, None)
    cdf = np.cumsum(counts) / max(np.sum(counts), 1e-9)
    idx = int(np.searchsorted(cdf, q))
    idx = min(max(idx, 0), len(edges) - 2)
    return float(edges[idx + 1])


# -----------------------------
# Privacy accountant (RDP composition if available)
# -----------------------------
class Accountant:
    def __init__(self, delta: float = 1e-6):
        self.delta = delta
        self.events = []  # (type, payload)
    def add_gaussian_event(self, noise_multiplier: float, sampling_probability: float):
        self.events.append(("gauss", (noise_multiplier, sampling_probability)))
    def add_pure_eps(self, eps: float):
        self.events.append(("pure", eps))
    def get_epsilon(self) -> float:
        pure = sum(payload for typ, payload in self.events if typ == "pure")
        if _HAS_DP_ACCOUNTING:
            orders = np.concatenate([np.linspace(1.25, 64, 200), np.array([128, 256])])
            total_rdp = np.zeros_like(orders, dtype=float)
            for typ, payload in self.events:
                if typ == "gauss":
                    noise_multiplier, q = payload
                    rdp = rdp_mod.compute_rdp(q=q, noise_multiplier=noise_multiplier, steps=1, orders=orders)
                    total_rdp += rdp
            if total_rdp.sum() > 0:
                eps_rdp, _, _ = rdp_mod.get_privacy_spent(orders=orders, rdp=total_rdp, delta=self.delta)
            else:
                eps_rdp = 0.0
            return float(eps_rdp + pure)
        else:
            gauss_eps = 0.0
            for typ, payload in self.events:
                if typ == "gauss":
                    noise_multiplier, q = payload
                    if noise_multiplier <= 0:
                        continue
                    gauss_eps += (q*q) / (2.0 * (noise_multiplier**2))
            return float(gauss_eps + pure)
    def reset(self):
        self.events = []


# -----------------------------
# Client residual computation
# -----------------------------

def client_compute_residual(model: nn.Module, data_loader: DataLoader, device: torch.device,
                            h_i_vec: np.ndarray, c_t_vec: np.ndarray, lr: float = 0.05, momentum: float = 0.9,
                            max_batches: int = 1) -> np.ndarray:
    local_model = copy.deepcopy(model).to(device)
    opt = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum)
    local_model.train()
    batches = 0
    for xb, yb in data_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = local_model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        batches += 1
        if batches >= max_batches:
            break
    g_i_vec = (flatten_params(local_model) - flatten_params(model)).detach().cpu().numpy().astype(np.float32)
    r_i = g_i_vec - h_i_vec
    return r_i


# -----------------------------
# Evaluation helpers within training module
# -----------------------------

def estimate_dp_noise_mse(sigma_eff: float, d: int) -> float:
    return float(d) * (sigma_eff ** 2)


def estimate_sketch_mse(x: np.ndarray, sketch: CountSketch) -> float:
    y = sketch.S(x)
    x_hat = sketch.unS(y)
    return float(np.mean((x_hat - x)**2))


# -----------------------------
# Transparency ledger I/O
# -----------------------------

def write_ledger_record(path: str, record_dict: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a") as f:
        f.write(json.dumps(record_dict) + "\n")


# -----------------------------
# Sigma calibration for a target budget
# -----------------------------

def calibrate_sigma_for_budget(T: int, q: float, eps_target: float, delta: float, iters: int = 30) -> float:
    acc = Accountant(delta=delta)
    def eps_for_sigma(sigma):
        acc.reset()
        for _ in range(T):
            acc.add_gaussian_event(noise_multiplier=sigma, sampling_probability=q)
        return acc.get_epsilon()
    lo, hi = 0.05, 100.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        e = eps_for_sigma(mid)
        if e > eps_target:
            lo = mid
        else:
            hi = mid
    return hi


# -----------------------------
# Experiments
# -----------------------------

def experiment1_pacer_vs_baselines(config: Dict[str, Any]):
    images_dir = config.get("images_dir", ".research/iteration1/images")
    models_dir = config.get("models_dir", "models")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("[Experiment 1] Utility–privacy under non-IID with PACER-FL vs baselines")
    set_seed(config.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_cuda", False) else "cpu")

    # Data selection (synthetic by default)
    use_synth = config.get("use_synthetic", True)
    num_classes = 10
    if use_synth:
        n_pc = config.get("synth_n_per_class", 30)
        img_size = config.get("synth_img_size", 8)
        train_ds = SyntheticImageDataset(n_per_class=n_pc, img_size=img_size, num_classes=num_classes, seed=config.get("seed", 0))
        test_ds = SyntheticImageDataset(n_per_class=max(10, n_pc//2), img_size=img_size, num_classes=num_classes, seed=config.get("seed", 1))
    else:
        if not _HAS_TORCHVISION:
            raise RuntimeError("torchvision not available; set use_synthetic=True for quick test.")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
        ])
        data_root = config.get("data_root", "./data")
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    train_labels = np.array([int(train_ds[i][1]) for i in range(len(train_ds))])

    N = config.get("N", 200)
    T = config.get("T", 20)
    n_t = config.get("n_t", 20)
    q = n_t / N
    batch_size = config.get("batch_size", 32)
    alpha_dir = config.get("alpha_dir", 0.1)

    client_indices_non_iid = dirichlet_noniid_split(train_labels, num_clients=N, alpha=alpha_dir, min_size=max(5, batch_size), seed=config.get("seed", 0))
    client_indices_iid = iid_split(len(train_labels), num_clients=N, min_size=max(5, batch_size), seed=config.get("seed", 1))

    global_model_pacer = SmallCIFAR10CNN().to(device)
    global_model_base = SmallCIFAR10CNN().to(device)
    d = sum(p.numel() for p in global_model_pacer.parameters())

    eps_total = config.get("eps", 2.0)
    delta = config.get("delta", 1e-6)
    eps_tree = 0.6 * eps_total
    eps_gate_total = 0.2 * eps_total
    eps_clip_total = 0.2 * eps_total
    eps_gate_round = eps_gate_total / T
    eps_clip_round = eps_clip_total / T

    sigma_base = calibrate_sigma_for_budget(T=T, q=q, eps_target=eps_total, delta=delta)
    print(f"[Experiment 1] Calibrated baseline i.i.d. Gaussian sigma={sigma_base:.3f} for eps≈{eps_total}")

    sigma_eff_global = calibrate_sigma_for_budget(T=T, q=q, eps_target=eps_tree, delta=delta)
    print(f"[Experiment 1] Calibrated PACER tree-effective sigma={sigma_eff_global:.3f} for eps_tree≈{eps_tree:.3f}")

    beta = config.get("beta", 0.1)
    alpha_cv = config.get("alpha_cv", 0.2)
    c_t_vec = np.zeros(d, dtype=np.float32)
    h_i = [np.zeros(d, dtype=np.float32) for _ in range(N)]
    L_min = config.get("L_min", max(1, int(0.8 * n_t)))
    rng = np.random.default_rng(config.get("seed", 0))

    vrf_seeds = {}
    target_var_per_node = {}
    for t in range(1, T+1):
        for node in tree_nodes_for_round(t):
            if node not in vrf_seeds:
                vrf_seeds[node] = int(rng.integers(1, 2**31-1))
            if node not in target_var_per_node:
                target_var_per_node[node] = 1.0

    ledger_path = config.get("ledger_path", os.path.join("data", "ledger_exp1.jsonl"))
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    if os.path.exists(ledger_path):
        os.remove(ledger_path)

    accs_pacer, accs_base = [], []
    losses_pacer, losses_base = [], []
    on_rates = []
    residual_norms_mean = []
    sketches_m = []

    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    eps_eff_round = eps_tree / T
    M_prev = 0.1

    for t in range(1, T+1):
        client_indices = client_indices_non_iid if (t % 2 == 1) else client_indices_iid
        cohort = rng.choice(N, size=n_t, replace=False)

        m_t = choose_m_t(d=d, n_t=n_t, eps_eff_t=eps_eff_round, M_t=M_prev, kappa=1.2)
        sketch_seed = int(rng.integers(1, 2**31-1))
        sketch = CountSketch(d=d, m=m_t, seed=sketch_seed)
        sketches_m.append(m_t)

        scale_factor = (sigma_eff_global**2) / max(1e-6, float(np.mean(list(target_var_per_node.values()))))
        target_var_scaled = {k: v*scale_factor for k, v in target_var_per_node.items()}

        per_client_norms = []
        client_loaders = {}
        for i in cohort:
            idxs = client_indices[i]
            client_loaders[i] = DataLoader(Subset(train_ds, idxs), batch_size=batch_size, shuffle=True)
            r_i = client_compute_residual(global_model_pacer, client_loaders[i], device, h_i[i], c_t_vec, max_batches=1)
            per_client_norms.append(np.linalg.norm(r_i))
        clip_max = float(np.percentile(per_client_norms, 99.0) + 1.0) if per_client_norms else 1.0
        C_res_t = private_quantile_from_histogram(np.array(per_client_norms, dtype=np.float32) if per_client_norms else np.array([0.0]), q=0.9, clip_max=clip_max, eps=eps_clip_round, bins=30, rng=rng)

        theta_t = 0.0 if t < T/2 else 0.05

        uploads = []
        ON_flags = []
        realized_ON = 0
        for i in cohort:
            r_i = client_compute_residual(global_model_pacer, client_loaders[i], device, h_i[i], c_t_vec, max_batches=1)
            norm = float(np.linalg.norm(r_i) + 1e-12)
            scale = min(1.0, C_res_t / norm) if C_res_t > 0 else 1.0
            r_i = (r_i * scale).astype(np.float32)
            cos_sim = float(np.dot(r_i, c_t_vec) / (np.linalg.norm(r_i)*np.linalg.norm(c_t_vec)+1e-12)) if np.linalg.norm(c_t_vec)>0 else 0.0
            s_i = max(cos_sim, 0.1 * norm / max(C_res_t, 1e-6))
            ON = dp_gate(s_i, theta_t, sensitivity=1.0, eps=eps_gate_round, rng=rng)
            ON_flags.append(ON)
            if not ON:
                r_i = np.zeros_like(r_i)
            y_i = sketch.S(r_i)
            shares_i = client_ddg_tree_shares(round_t=t, sketch_dim=m_t, seeds=vrf_seeds, target_var_per_node=target_var_scaled, L_min=L_min, cohort_size=n_t)
            uploads.append((y_i.astype(np.float32), shares_i))
            realized_ON += int(ON)
            h_i[i] = (1.0 - 0.2) * h_i[i] + 0.2 * c_t_vec

        Y_sum = np.sum([u[0] for u in uploads], axis=0) if uploads else np.zeros(m_t, dtype=np.float32)
        Shares_sum = np.sum([u[1] for u in uploads], axis=0).astype(np.float32) if uploads else np.zeros(m_t, dtype=np.float32)
        sketch_agg = Y_sum + Shares_sum
        agg_update = sketch.unS(sketch_agg)
        denom = max(realized_ON, 1)
        vec = flatten_params(global_model_pacer).cpu().numpy()
        vec_new = vec + agg_update / float(denom)
        assign_params(global_model_pacer, torch.tensor(vec_new, dtype=torch.float32))
        c_t_vec = (1.0 - beta) * c_t_vec + beta * (agg_update / float(denom))

        acc_pacer = evaluate_model(global_model_pacer, test_loader, device)
        accs_pacer.append(acc_pacer)
        on_rate = sum(ON_flags) / len(ON_flags) if ON_flags else 0.0
        on_rates.append(on_rate)
        residual_norms_mean.append(float(np.mean(per_client_norms)) if per_client_norms else 0.0)
        losses_pacer.append(float(np.mean(per_client_norms)) if per_client_norms else 0.0)  # proxy

        # Baseline (FedAvg + server i.i.d Gaussian noise)
        updates = []
        for i in cohort:
            local_model = copy.deepcopy(global_model_base).to(device)
            opt = torch.optim.SGD(local_model.parameters(), lr=0.05, momentum=0.9)
            for xb, yb in client_loaders[i]:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = local_model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
                break
            update_vec = (flatten_params(local_model) - flatten_params(global_model_base)).detach().cpu().numpy().astype(np.float32)
            updates.append(update_vec)
        if updates:
            mean_update = np.mean(np.stack(updates, axis=0), axis=0)
        else:
            mean_update = np.zeros(d, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=sigma_base, size=d).astype(np.float32)
        vec_b = flatten_params(global_model_base).cpu().numpy()
        vec_b_new = vec_b + mean_update + noise
        assign_params(global_model_base, torch.tensor(vec_b_new, dtype=torch.float32))
        acc_base = evaluate_model(global_model_base, test_loader, device)
        accs_base.append(acc_base)
        losses_base.append(float(np.mean([np.linalg.norm(u) for u in updates])) if updates else 0.0)

        write_ledger_record(ledger_path, {
            "round": t,
            "vrf_seeds": {str(k): int(v) for k, v in vrf_seeds.items()},
            "target_var_per_node": {str(k): float(v) for k, v in target_var_scaled.items()},
            "L_min": int(L_min),
            "realized_cohort": int(n_t),
            "sketch_seed": int(sketch_seed),
            "sketch_m": int(m_t),
            "clip_quantiles": {"global": float(C_res_t)},
            "gate_theta": float(theta_t),
            "eps_eff_round": float(eps_eff_round),
            "eps_gate_round": float(eps_gate_round),
            "eps_clip_round": float(eps_clip_round),
            "restart_mode": "NoTreeRestart"
        })

        print(f"Round {t:03d}: PACER acc={acc_pacer:.2f}%, Base acc={acc_base:.2f}%, ON-rate={on_rate:.2f}, m_t={m_t}")

    # Save models
    torch.save(global_model_pacer.state_dict(), os.path.join(models_dir, "pacer_exp1_final.pt"))
    torch.save(global_model_base.state_dict(), os.path.join(models_dir, "base_exp1_final.pt"))

    # Plots
    plt.figure(figsize=(6,4))
    plt.plot(range(1, T+1), accs_pacer, label="PACER-FL")
    plt.plot(range(1, T+1), accs_base, label="DP-FedAvg (i.i.d.)")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.title("Accuracy vs Rounds")
    plt.savefig(os.path.join(images_dir, "accuracy_pacerfl_vs_baselines.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(range(1, T+1), losses_pacer, label="PACER proxy loss (residual norm)")
    plt.plot(range(1, T+1), losses_base, label="Baseline proxy loss")
    plt.xlabel("Round")
    plt.ylabel("Proxy Loss / Norm")
    plt.legend()
    plt.title("Training proxy loss vs Rounds")
    plt.savefig(os.path.join(images_dir, "training_loss_pacerfl_vs_baselines.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(range(1, T+1), residual_norms_mean, label="Mean residual norm")
    plt.xlabel("Round")
    plt.ylabel("Mean ||r_i||")
    plt.legend()
    plt.title("Residual norm trajectory")
    plt.savefig(os.path.join(images_dir, "residual_norms_pacerfl.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(range(1, T+1), on_rates, label="ON-rate")
    plt.xlabel("Round")
    plt.ylabel("ON-rate")
    plt.legend()
    plt.title("DP gating ON-rate")
    plt.savefig(os.path.join(images_dir, "on_rate_pacerfl.pdf"), bbox_inches="tight")
    plt.close()

    print("[Experiment 1] Completed. Figures saved to", images_dir)

    return {
        "accs_pacer": accs_pacer,
        "accs_base": accs_base,
        "sketch_dims": sketches_m,
        "ledger_path": ledger_path
    }


def experiment2_comm_efficiency(config: Dict[str, Any]):
    images_dir = config.get("images_dir", ".research/iteration1/images")
    models_dir = config.get("models_dir", "models")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("[Experiment 2] Communication–efficiency with FO-aware compression + DP gating")
    set_seed(config.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_cuda", False) else "cpu")

    use_synth = config.get("use_synthetic", True)
    num_classes = 10
    if use_synth:
        train_ds = SyntheticImageDataset(n_per_class=config.get("synth_n_per_class", 20), img_size=config.get("synth_img_size", 8), num_classes=num_classes, seed=config.get("seed", 0))
        test_ds = SyntheticImageDataset(n_per_class=config.get("synth_n_per_class", 10), img_size=config.get("synth_img_size", 8), num_classes=num_classes, seed=config.get("seed", 1))
    else:
        if not _HAS_TORCHVISION:
            raise RuntimeError("torchvision not available; set use_synthetic=True for quick test.")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
        ])
        data_root = config.get("data_root", "./data")
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    labels = np.array([int(train_ds[i][1]) for i in range(len(train_ds))])

    N = config.get("N", 100)
    T = config.get("T", 15)
    n_t = config.get("n_t", 10)
    q = n_t / N
    batch_size = config.get("batch_size", 32)

    client_indices = dirichlet_noniid_split(labels, num_clients=N, alpha=config.get("alpha_dir", 0.1), min_size=max(5, batch_size), seed=config.get("seed", 0))

    model_pacer = SmallCIFAR10CNN().to(device)
    d = sum(p.numel() for p in model_pacer.parameters())

    eps_total = config.get("eps", 2.0)
    delta = config.get("delta", 1e-6)
    eps_tree = 0.6 * eps_total
    eps_gate_round = (0.2 * eps_total) / T
    eps_clip_round = (0.2 * eps_total) / T
    sigma_eff_global = calibrate_sigma_for_budget(T=T, q=q, eps_target=eps_tree, delta=delta)

    c_t_vec = np.zeros(d, dtype=np.float32)
    h_i = [np.zeros(d, dtype=np.float32) for _ in range(N)]
    rng = np.random.default_rng(config.get("seed", 0))

    vrf_seeds = {}
    target_var_per_node = {}
    for t in range(1, T+1):
        for node in tree_nodes_for_round(t):
            if node not in vrf_seeds:
                vrf_seeds[node] = int(rng.integers(1, 2**31-1))
            if node not in target_var_per_node:
                target_var_per_node[node] = 1.0
    scale_factor = (sigma_eff_global**2) / max(1e-6, float(np.mean(list(target_var_per_node.values()))))
    target_var_scaled = {k: v*scale_factor for k, v in target_var_per_node.items()}

    bits_per_round_pacer = []
    bits_per_round_pacer_wo_gate = []
    accs_pacer = []
    align_ratios = []

    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    eps_eff_round = eps_tree / T
    M_prev = 0.1
    L_min = max(1, int(0.8 * n_t))

    for t in range(1, T+1):
        cohort = rng.choice(N, size=n_t, replace=False)
        m_t = choose_m_t(d=d, n_t=n_t, eps_eff_t=eps_eff_round, M_t=M_prev, kappa=1.2)
        sketch_seed = int(rng.integers(1, 2**31-1))
        sketch = CountSketch(d=d, m=m_t, seed=sketch_seed)

        norms = []
        loaders = {}
        for i in cohort:
            idxs = client_indices[i]
            loaders[i] = DataLoader(Subset(train_ds, idxs), batch_size=batch_size, shuffle=True)
            r_i = client_compute_residual(model_pacer, loaders[i], device, h_i[i], c_t_vec, max_batches=1)
            norms.append(np.linalg.norm(r_i))
        clip_max = float(np.percentile(norms, 99.0) + 1.0) if norms else 1.0
        C_res_t = private_quantile_from_histogram(np.array(norms, dtype=np.float32) if norms else np.array([0.0]), q=0.9, clip_max=clip_max, eps=eps_clip_round, bins=30, rng=rng)

        theta_t = 0.0 if t < T/2 else 0.05
        uploads_on = []
        uploads_always = []
        realized_ON = 0
        for i in cohort:
            r_i = client_compute_residual(model_pacer, loaders[i], device, h_i[i], c_t_vec, max_batches=1)
            norm = float(np.linalg.norm(r_i) + 1e-12)
            scale = min(1.0, C_res_t / norm) if C_res_t > 0 else 1.0
            r_i = (r_i * scale).astype(np.float32)
            cos_sim = float(np.dot(r_i, c_t_vec) / (np.linalg.norm(r_i)*np.linalg.norm(c_t_vec)+1e-12)) if np.linalg.norm(c_t_vec)>0 else 0.0
            s_i = max(cos_sim, 0.1 * norm / max(C_res_t, 1e-6))
            ON = dp_gate(s_i, theta_t, sensitivity=1.0, eps=eps_gate_round, rng=rng)
            y_i = sketch.S(r_i if ON else np.zeros_like(r_i))
            shares_i = client_ddg_tree_shares(round_t=t, sketch_dim=m_t, seeds=vrf_seeds, target_var_per_node=target_var_scaled, L_min=L_min, cohort_size=n_t)
            uploads_on.append((y_i.astype(np.float32), shares_i))
            y_i_full = sketch.S(r_i)
            uploads_always.append((y_i_full.astype(np.float32), shares_i))
            realized_ON += int(ON)
            h_i[i] = 0.8 * h_i[i] + 0.2 * c_t_vec

        Ysum = np.sum([u[0] for u in uploads_on], axis=0) if uploads_on else np.zeros(m_t, dtype=np.float32)
        Zsum = np.sum([u[1] for u in uploads_on], axis=0).astype(np.float32) if uploads_on else np.zeros(m_t, dtype=np.float32)
        agg = sketch.unS(Ysum + Zsum)
        denom = max(realized_ON, 1)
        vec = flatten_params(model_pacer).cpu().numpy()
        vec_new = vec + agg / float(denom)
        assign_params(model_pacer, torch.tensor(vec_new, dtype=torch.float32))
        c_t_vec = 0.9 * c_t_vec + 0.1 * (agg / float(denom))

        bits_pacer = bits_per_client_countsketch(m_t)
        bits_per_round_pacer.append(bits_pacer)
        bits_per_round_pacer_wo_gate.append(bits_per_client_countsketch(m_t))

        probe = np.random.normal(0, 1, size=d).astype(np.float32)
        mse_sketch = estimate_sketch_mse(probe, sketch)
        mse_dp = estimate_dp_noise_mse(sigma_eff_global, d)
        align = mse_sketch / max(mse_dp, 1e-9)
        align_ratios.append(float(align))

        acc = evaluate_model(model_pacer, test_loader, device)
        accs_pacer.append(acc)
        print(f"Round {t:03d}: bits/client={bits_pacer}, align={align:.3e}, acc={acc:.2f}%")

    torch.save(model_pacer.state_dict(), os.path.join(models_dir, "pacer_exp2_final.pt"))

    plt.figure(figsize=(6,4))
    plt.plot(range(1, T+1), bits_per_round_pacer, label="PACER-FL (FO-aware)")
    plt.plot(range(1, T+1), bits_per_round_pacer_wo_gate, label="PACER-wo-gate")
    plt.xlabel("Round")
    plt.ylabel("Bits per client")
    plt.legend()
    plt.title("Communication per round")
    plt.savefig(os.path.join(images_dir, "uplink_bits_pacerfl.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(range(1, T+1), align_ratios, label="Alignment ratio")
    plt.axhline(1.0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("Round")
    plt.ylabel("Sketch MSE / DP noise MSE")
    plt.legend()
    plt.title("FO alignment ratio")
    plt.savefig(os.path.join(images_dir, "alignment_ratio_pacerfl.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(range(1, T+1), accs_pacer, label="PACER-FL acc")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy vs Rounds (PACER-FL)")
    plt.savefig(os.path.join(images_dir, "accuracy_pacerfl.pdf"), bbox_inches="tight")
    plt.close()

    print("[Experiment 2] Completed. Figures saved to", images_dir)
    return {
        "bits_pacer": bits_per_round_pacer,
        "align_ratios": align_ratios,
        "accs_pacer": accs_pacer
    }


def tree_restart_schedule(T: int, mode: str = "NoTreeRestart", checkpoints=None):
    if mode == "NoTreeRestart":
        return {t: tree_nodes_for_round(t) for t in range(1, T+1)}
    if mode == "SometimesRestart":
        if checkpoints is None:
            checkpoints = [int(T*0.33), int(T*0.66)]
        mapping = {}
        start = 1
        for cp in checkpoints + [T+1]:
            for t in range(start, cp):
                nodes = []
                k, x = 0, (t-start+1)
                while x > 0:
                    if x & 1:
                        nodes.append((k, start + ((t-start+1) >> k) << k))
                    k += 1
                    x >>= 1
                mapping[t] = nodes
            start = cp
        return mapping
    if mode == "TreeRestart":
        period = max(1, int(math.sqrt(T)))
        mapping = {}
        for t in range(1, T+1):
            start_block = ((t-1)//period)*period + 1
            x = (t - start_block + 1)
            nodes = []
            k = 0
            while x > 0:
                if x & 1:
                    nodes.append((k, start_block + ((t-start_block+1)>>k)<<k))
                k += 1
                x >>= 1
            mapping[t] = nodes
        return mapping
    raise ValueError("Unknown mode")


def auditor_verify_ledger(ledger_path: str) -> bool:
    if not os.path.exists(ledger_path):
        print("Ledger not found:", ledger_path)
        return False
    recs = read_ledger(ledger_path)
    ok = True
    for rec in recs:
        t = rec["round"]; m = rec["sketch_m"]
        from ast import literal_eval
        seeds = {literal_eval(k): v for k, v in rec["vrf_seeds"].items()}
        vars_node = {literal_eval(k): v for k, v in rec["target_var_per_node"].items()}
        agg_noise = auditor_reconstruct_noise(t, m, seeds, vars_node, rec["L_min"], rec["realized_cohort"]).astype(np.float32)
        var_emp = float(np.var(agg_noise))
        var_target_mean = float(np.mean(list(vars_node.values()))) if vars_node else 0.0
        if var_target_mean > 0 and var_emp + 1e-6 < 0.5 * var_target_mean:
            print(f"[Auditor] Round {t}: empirical noise var {var_emp:.4f} < 0.5x target mean {var_target_mean:.4f}")
            ok = False
    return ok


def experiment3_auditability(config: Dict[str, Any]):
    images_dir = config.get("images_dir", ".research/iteration1/images")
    models_dir = config.get("models_dir", "models")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("[Experiment 3] Auditability & robustness under dropouts and restarts")
    set_seed(config.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_cuda", False) else "cpu")

    train_ds = SyntheticImageDataset(n_per_class=config.get("synth_n_per_class", 20), img_size=config.get("synth_img_size", 8), num_classes=10, seed=config.get("seed", 0))
    test_ds = SyntheticImageDataset(n_per_class=config.get("synth_n_per_class", 10), img_size=config.get("synth_img_size", 8), num_classes=10, seed=config.get("seed", 1))
    labels = np.array([int(train_ds[i][1]) for i in range(len(train_ds))])

    N = config.get("N", 80)
    n_t = config.get("n_t", 8)
    T = config.get("T", 12)
    q = n_t / N
    batch_size = config.get("batch_size", 32)

    client_indices = dirichlet_noniid_split(labels, num_clients=N, alpha=config.get("alpha_dir", 0.1), min_size=max(5, batch_size), seed=config.get("seed", 0))

    model = SmallCIFAR10CNN().to(device)
    d = sum(p.numel() for p in model.parameters())

    eps_total = config.get("eps", 2.0)
    delta = config.get("delta", 1e-6)
    eps_tree = 0.6 * eps_total
    eps_gate_round = (0.2 * eps_total) / T
    eps_clip_round = (0.2 * eps_total) / T
    sigma_eff_global = calibrate_sigma_for_budget(T=T, q=q, eps_target=eps_tree, delta=delta)

    rng = np.random.default_rng(config.get("seed", 0))

    restart_mode = config.get("restart_mode", "SometimesRestart")
    sched = tree_restart_schedule(T, mode=restart_mode)

    vrf_seeds = {}
    target_var_per_node = {}
    for t in range(1, T+1):
        for node in sched[t]:
            if node not in vrf_seeds:
                vrf_seeds[node] = int(rng.integers(1, 2**31-1))
            if node not in target_var_per_node:
                target_var_per_node[node] = 1.0
    scale_factor = (sigma_eff_global**2) / max(1e-6, float(np.mean(list(target_var_per_node.values())))) if target_var_per_node else 1.0
    target_var_scaled = {k: v*scale_factor for k, v in target_var_per_node.items()}

    ledger_path = config.get("ledger_path", os.path.join("data", f"ledger_exp3_{restart_mode}.jsonl"))
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    if os.path.exists(ledger_path):
        os.remove(ledger_path)

    regime = config.get("regime", "R3")  # R1 mild, R2 heavy, R3 bursty
    if regime == "R1":
        p_dropout = 0.2
    elif regime == "R2":
        p_dropout = 0.6
    else:
        p_dropout = None

    L_min = max(1, int(0.8 * n_t))
    c_t_vec = np.zeros(d, dtype=np.float32)
    h_i = [np.zeros(d, dtype=np.float32) for _ in range(N)]

    accs = []
    realized_cohorts = []

    for t in range(1, T+1):
        if regime == "R3":
            p = 0.2 if (t % 4 in [1,2]) else 0.8
        else:
            p = p_dropout
        base = rng.choice(N, size=n_t, replace=False)
        mask = rng.random(size=n_t) > p
        cohort = base[mask]
        realized_L = len(cohort)
        realized_cohorts.append(realized_L)
        if realized_L == 0:
            cohort = rng.choice(N, size=max(1, n_t//2), replace=False)
            realized_L = len(cohort)

        eps_eff_round = eps_tree / T
        m_t = choose_m_t(d=d, n_t=max(realized_L, 1), eps_eff_t=eps_eff_round, M_t=0.1, kappa=1.2)
        sketch_seed = int(rng.integers(1, 2**31-1))
        sketch = CountSketch(d=d, m=m_t, seed=sketch_seed)

        norms = []
        loaders = {}
        for i in cohort:
            idxs = client_indices[i]
            loaders[i] = DataLoader(Subset(train_ds, idxs), batch_size=batch_size, shuffle=True)
            r_i = client_compute_residual(model, loaders[i], device, h_i[i], c_t_vec, max_batches=1)
            norms.append(np.linalg.norm(r_i))
        clip_max = float(np.percentile(norms, 99.0)+1.0) if norms else 1.0
        C_res_t = private_quantile_from_histogram(np.array(norms, dtype=np.float32) if norms else np.array([0.0]), q=0.9, clip_max=clip_max, eps=eps_clip_round, bins=30, rng=rng)

        theta_t = 0.0
        uploads = []
        realized_ON = 0
        for i in cohort:
            r_i = client_compute_residual(model, loaders[i], device, h_i[i], c_t_vec, max_batches=1)
            norm = float(np.linalg.norm(r_i) + 1e-12)
            scale = min(1.0, C_res_t / norm) if C_res_t > 0 else 1.0
            r_i = (r_i * scale).astype(np.float32)
            cos_sim = float(np.dot(r_i, c_t_vec) / (np.linalg.norm(r_i)*np.linalg.norm(c_t_vec)+1e-12)) if np.linalg.norm(c_t_vec)>0 else 0.0
            s_i = max(cos_sim, 0.1 * norm / max(C_res_t, 1e-6))
            ON = dp_gate(s_i, theta_t, sensitivity=1.0, eps=eps_gate_round, rng=rng)
            y_i = sketch.S(r_i if ON else np.zeros_like(r_i))
            shares_i = client_ddg_tree_shares(round_t=t, sketch_dim=m_t, seeds=vrf_seeds, target_var_per_node=target_var_scaled, L_min=L_min, cohort_size=max(realized_L, 1))
            uploads.append((y_i.astype(np.float32), shares_i))
            realized_ON += int(ON)
            h_i[i] = 0.8 * h_i[i] + 0.2 * c_t_vec

        Ysum = np.sum([u[0] for u in uploads], axis=0) if uploads else np.zeros(m_t, dtype=np.float32)
        Zsum = np.sum([u[1] for u in uploads], axis=0).astype(np.float32) if uploads else np.zeros(m_t, dtype=np.float32)
        agg = sketch.unS(Ysum + Zsum)
        denom = max(realized_ON, 1)
        vec = flatten_params(model).cpu().numpy()
        vec_new = vec + agg / float(denom)
        assign_params(model, torch.tensor(vec_new, dtype=torch.float32))
        c_t_vec = 0.9 * c_t_vec + 0.1 * (agg / float(denom))

        write_ledger_record(ledger_path, {
            "round": t,
            "vrf_seeds": {str(k): int(v) for k, v in vrf_seeds.items()},
            "target_var_per_node": {str(k): float(target_var_scaled.get(k, 0.0)) for k in target_var_scaled},
            "L_min": int(L_min),
            "realized_cohort": int(realized_L),
            "sketch_seed": int(sketch_seed),
            "sketch_m": int(m_t),
            "clip_quantiles": {"global": float(C_res_t)},
            "gate_theta": float(theta_t),
            "eps_eff_round": float(eps_eff_round),
            "eps_gate_round": float(eps_gate_round),
            "eps_clip_round": float(eps_clip_round),
            "restart_mode": restart_mode
        })

        acc = evaluate_model(model, DataLoader(test_ds, batch_size=128, shuffle=False), device)
        accs.append(acc)
        print(f"Round {t:03d} [{regime}/{restart_mode}]: realized_L={realized_L}, acc={acc:.2f}%")

    torch.save(model.state_dict(), os.path.join(models_dir, "pacer_exp3_final.pt"))

    ok = auditor_verify_ledger(ledger_path)
    print("[Experiment 3] Auditor verification:", "PASS" if ok else "FAIL")

    plt.figure(figsize=(6,4))
    plt.plot(range(1, T+1), realized_cohorts, label="Realized cohort size")
    plt.axhline(L_min, color='r', linestyle='--', label='L_min')
    plt.xlabel("Round")
    plt.ylabel("Cohort size")
    plt.legend()
    plt.title("Cohort sizes and L_min")
    plt.savefig(os.path.join(images_dir, "cohort_sizes_robustness.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(range(1, T+1), accs, label="Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy under restarts/dropouts")
    plt.savefig(os.path.join(images_dir, "accuracy_robustness.pdf"), bbox_inches="tight")
    plt.close()

    recs = read_ledger(ledger_path)
    if recs:
        rec = recs[-1]
        from ast import literal_eval
        m = rec["sketch_m"]
        seeds = {literal_eval(k): v for k, v in rec["vrf_seeds"].items()}
        vars_node = {literal_eval(k): v for k, v in rec["target_var_per_node"].items()}
        agg_noise = auditor_reconstruct_noise(rec["round"], m, seeds, vars_node, rec["L_min"], rec["realized_cohort"]).astype(np.float32)
        plt.figure(figsize=(6,4))
        sns.histplot(agg_noise, bins=30, stat='density')
        plt.title("Auditor reconstructed noise (sketch domain)")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.savefig(os.path.join(images_dir, "audit_noise_hist.pdf"), bbox_inches="tight")
        plt.close()

    print("[Experiment 3] Completed. Figures saved to", images_dir)
    return {"auditor_ok": ok, "ledger_path": ledger_path}


# -----------------------------
# Quick functional test (fast, CPU-friendly)
# -----------------------------

def run_quick_test():
    images_dir = ".research/iteration1/images"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    cfg1 = {
        "seed": 1, "use_cuda": False, "use_synthetic": True,
        "synth_n_per_class": 10, "synth_img_size": 8,
        "N": 40, "T": 8, "n_t": 8, "batch_size": 16,
        "alpha_dir": 0.3, "eps": 1.5, "delta": 1e-6,
        "ledger_path": os.path.join("data", "ledger_quick_exp1.jsonl"),
        "images_dir": images_dir, "models_dir": "models"
    }
    res1 = experiment1_pacer_vs_baselines(cfg1)

    cfg2 = {
        "seed": 2, "use_cuda": False, "use_synthetic": True,
        "synth_n_per_class": 10, "synth_img_size": 8,
        "N": 40, "T": 8, "n_t": 8, "batch_size": 16,
        "alpha_dir": 0.1, "eps": 1.5, "delta": 1e-6,
        "images_dir": images_dir, "models_dir": "models"
    }
    res2 = experiment2_comm_efficiency(cfg2)

    cfg3 = {
        "seed": 3, "use_cuda": False, "synth_n_per_class": 10, "synth_img_size": 8,
        "N": 40, "T": 8, "n_t": 8, "batch_size": 16,
        "alpha_dir": 0.1, "eps": 1.5, "delta": 1e-6,
        "restart_mode": "SometimesRestart", "regime": "R3",
        "ledger_path": os.path.join("data", "ledger_quick_exp3.jsonl"),
        "images_dir": images_dir, "models_dir": "models"
    }
    res3 = experiment3_auditability(cfg3)

    print("Quick test results summary:")
    print(" - Exp1: acc_end_pacer=%.2f, acc_end_base=%.2f, ledger=%s" % (res1["accs_pacer"][-1], res1["accs_base"][-1], res1["ledger_path"]))
    print(" - Exp2: last_bits=%d, last_align=%.3e, last_acc=%.2f" % (res2["bits_pacer"][-1], res2["align_ratios"][-1], res2["accs_pacer"][-1]))
    print(" - Exp3: auditor=", "PASS" if res3["auditor_ok"] else "FAIL", ", ledger=", res3["ledger_path"]) 
    print("All figures saved as .pdf to:", images_dir)


if __name__ == "__main__":
    run_quick_test()
