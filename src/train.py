#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train module for One-Pass Sketch-and-Validate (OP-S&V)

Implements the core algorithm (head-only version), baselines, and training loops.
This module is orchestrated by src/main.py, which also calls evaluate.py to create
paper-ready figures saved as PDFs in .research/iteration1/images.

Key components included here:
- CountSketch hashing and Robust Frequent Directions (RFD) with exponential forgetting
- Validation-aligned gradient projection and a simplified online dual controller
- Nested reservoir sampling providing budget-nested subsets
- Training utilities for linear heads on frozen features
- Baseline selectors (random, loss, entropy, EL2N, GraNd, optional herding when sklearn is available)

The code is designed to run on NVIDIA Tesla T4 16GB, with CPU fallback.
"""

import os
import time
import math
import json
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .preprocess import FeatureDataset
from . import evaluate as eval_utils

try:
    from sklearn.metrics import confusion_matrix
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# ------------------------------
# Reproducibility and device
# ------------------------------

def set_seed(seed: int = 13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: Optional[str] = None):
    if device_str in (None, "auto"):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device_str)


# ------------------------------
# CountSketch and hashing utils
# ------------------------------

def create_s_idx_sign(P: int, d_hat: int, seed: int = 13, device=None):
    rng = np.random.RandomState(seed)
    s_idx = torch.tensor(rng.randint(0, d_hat, size=(P,), dtype=np.int64), dtype=torch.long, device=device)
    s_sign = torch.tensor(rng.choice([-1.0, 1.0], size=(P,), replace=True), dtype=torch.float32, device=device)
    return s_idx, s_sign


def countsketch(M: torch.Tensor, s_idx: torch.Tensor, s_sign: torch.Tensor, d_hat: int) -> torch.Tensor:
    # M: [B, P]
    B, P = M.shape
    out = M.new_zeros(B, d_hat)
    out.index_add_(1, s_idx, M * s_sign)  # broadcast along batch dimension
    return out


# ------------------------------
# Robust Frequent Directions (RFD) with forgetting (simplified)
# ------------------------------

class RFD:
    def __init__(self, d_hat: int, r_max: int = 64, forgetting: float = 0.98, device=None):
        self.d_hat = d_hat
        self.r_max = r_max
        self.forgetting = forgetting
        self.device = device or get_device()
        self.B = torch.zeros(d_hat, r_max, device=self.device)
        self.rank = 0
        self.prev_proj = None

    @torch.no_grad()
    def update(self, X: torch.Tensor) -> Tuple[float, float]:
        # X: [B, d_hat]
        self.B.mul_(math.sqrt(self.forgetting))
        Y = torch.cat([self.B, X.T], dim=1)  # [d_hat, r_max + B]
        Q, R = torch.linalg.qr(Y, mode='reduced')
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
        k = min(self.r_max, U.size(1))
        S2 = S[:k]**2
        tau = S2[k-1].item() if k > 0 else 0.0
        S_shrink = torch.clamp(S2 - tau, min=0.0).sqrt()
        keep = (S_shrink > 1e-6)
        k_new = int(keep.sum().item())
        if k_new == 0:
            return 0.0, 0.0
        self.rank = k_new
        self.B[:, :k_new] = (Q @ U[:, :k_new]) * S_shrink[:k_new]
        # Δ_spec approximation via projector difference
        U_basis, _ = torch.linalg.qr(self.B[:, :k_new], mode='reduced')
        proj = U_basis @ U_basis.T
        if self.prev_proj is None:
            delta = 0.0
        else:
            delta = torch.linalg.norm(proj - self.prev_proj, ord='fro').item()
        self.prev_proj = proj
        tail = (S_shrink[k_new:]**2).sum().item() if k_new < len(S_shrink) else 0.0
        return tail, delta

    def basis(self) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros(self.d_hat, 1, device=self.B.device)
        Q, _ = torch.linalg.qr(self.B[:, :self.rank], mode='reduced')
        return Q


# ------------------------------
# Nested reservoir sampling with fixed keys
# ------------------------------

class NestedReservoir:
    def __init__(self, max_k: int, seed: int = 13):
        self.max_k = max_k
        self._heap = []  # stores (-priority, idx, weight)
        self.seed = seed
        self.rng = random.Random(seed)

    def push(self, idx: int, weight: float, priority: float):
        import heapq
        item = (-float(priority), int(idx), float(weight))
        if len(self._heap) < self.max_k:
            heapq.heappush(self._heap, item)
        else:
            if item > self._heap[0]:
                heapq.heapreplace(self._heap, item)

    def topk(self, k: int) -> List[Tuple[int, float]]:
        items = sorted(self._heap)  # ascending by -priority -> descending by priority
        return [(idx, w) for (_, idx, w) in items[:k]]

    def size(self) -> int:
        return len(self._heap)

    def merge(self, other: 'NestedReservoir') -> 'NestedReservoir':
        # priority-union merge (keeps highest-priority items)
        import heapq
        merged = NestedReservoir(max_k=self.max_k, seed=self.seed)
        heap = self._heap + other._heap
        heapq.heapify(heap)
        while len(merged._heap) < merged.max_k and len(heap) > 0:
            merged._heap.append(heapq.heappop(heap))
        return merged


def verify_nestedness(budget_indices: Dict[float, List[int]]) -> float:
    budgets = sorted(budget_indices.keys())
    total_checks, ok = 0, 0
    prev_set = None
    for b in budgets:
        cur = set(budget_indices[b])
        if prev_set is not None:
            total_checks += 1
            if prev_set.issubset(cur):
                ok += 1
        prev_set = cur
    return (ok / total_checks) if total_checks > 0 else 1.0


# ------------------------------
# Gradients, Fisher, validation gradients
# ------------------------------

def per_sample_head_residual(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    y_onehot = F.one_hot(y, logits.size(1)).float()
    return p - y_onehot


def flatten_head_grads(residuals: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    return torch.einsum('bc,bd->bcd', residuals, features).reshape(features.size(0), -1)


def compute_fisher_diag(loader: DataLoader, head: nn.Linear) -> torch.Tensor:
    device = next(head.parameters()).device
    P = head.weight.numel()
    fisher = torch.zeros(P, device=device)
    count = 0
    for batch in loader:
        if len(batch) == 3:
            Xb, yb, _ = batch
        else:
            Xb, yb, _ = batch[0], batch[1], batch[2]
        Xb = Xb.to(device); yb = yb.to(device)
        logits = Xb @ head.weight.T
        res = per_sample_head_residual(logits, yb)
        g = flatten_head_grads(res, Xb)
        fisher += (g * g).mean(dim=0)
        count += 1
    if count > 0:
        fisher /= float(count)
    return fisher


def validation_gradient(loader: DataLoader, head: nn.Linear, objective: str = 'acc', temperature: float = 1.0) -> torch.Tensor:
    device = next(head.parameters()).device
    head.zero_grad(set_to_none=True)
    total_loss = 0.0
    for batch in loader:
        if len(batch) == 3:
            Xb, yb, _ = batch
        else:
            Xb, yb, _ = batch[0], batch[1], batch[2]
        Xb = Xb.to(device); yb = yb.to(device)
        logits = Xb @ head.weight.T
        if objective in ['acc', 'cal']:
            logits_scaled = logits / (temperature if objective == 'cal' else 1.0)
            loss = F.cross_entropy(logits_scaled, yb, reduction='sum')
        else:
            raise ValueError(f"Unknown objective for validation_gradient: {objective}")
        total_loss = total_loss + loss
    total_loss.backward()
    v = head.weight.grad.view(-1).detach()
    head.zero_grad(set_to_none=True)
    return v


# ------------------------------
# OP-S&V selection (head-only, acc+cal objectives)
# ------------------------------

def op_snv_select(
    stream_loader: DataLoader,
    head: nn.Linear,
    d_hat: int,
    fisher_diag: torch.Tensor,
    v_acc: torch.Tensor,
    v_cal: torch.Tensor,
    s_idx: torch.Tensor,
    s_sign: torch.Tensor,
    budgets: Tuple[float, ...],
    gamma: float = 5e-4,
    forgetting: float = 0.98,
    r_max: int = 64,
    tau_acc: float = -0.01,
    tau_cal: float = -0.005,
    seed: int = 13,
) -> Tuple[NestedReservoir, float, int, Dict[str, List[float]]]:
    device = next(head.parameters()).device
    rfd = RFD(d_hat=d_hat, r_max=r_max, forgetting=forgetting, device=device)
    P_inv = 1.0 / (fisher_diag + 1e-6)

    def proj_vec(v: torch.Tensor) -> torch.Tensor:
        vP = v * P_inv
        v_cs = countsketch(vP[None, :], s_idx, s_sign, d_hat)[0]
        U = rfd.basis()
        return U @ (U.T @ v_cs)

    v_acc_proj = proj_vec(v_acc)
    v_cal_proj = proj_vec(v_cal)
    nA = v_acc_proj.norm() + 1e-8
    nC = v_cal_proj.norm() + 1e-8

    lam_acc, lam_cal = 1.0, 1.0
    ema_acc, ema_cal = 0.0, 0.0
    var_acc, var_cal = 0.0, 0.0
    beta = 0.9

    N_stream = len(stream_loader.dataset)
    max_k = int(max(budgets) * max(1, N_stream))
    reservoir = NestedReservoir(max_k=max_k, seed=seed)

    start = time.time()
    peak_mem = 0
    diag = {"lambda_acc": [], "lambda_cal": [], "delta_spec": [], "spectral_tail": []}

    for batch in stream_loader:
        Xb, yb, idxb = batch[0].to(device), batch[1].to(device), batch[2]
        logits = Xb @ head.weight.T
        res = per_sample_head_residual(logits, yb)
        g_flat = flatten_head_grads(res, Xb)
        g_cs = countsketch(g_flat, s_idx, s_sign, d_hat)
        tail, delta = rfd.update(g_cs)
        U = rfd.basis()
        proj = g_cs @ U @ U.T
        novelty = (g_cs - proj).pow(2).sum(dim=1)
        acc_imp = torch.clamp(-(g_cs @ v_acc_proj) / nA, min=0)
        cal_imp = torch.clamp(-(g_cs @ v_cal_proj) / nC, min=0)
        score = novelty + lam_acc * acc_imp + lam_cal * cal_imp
        for s, idxi in zip(score, idxb):
            key = (hash(int(idxi)) & 0xffffffff) / 2**32
            priority = float((s.item() + 1e-8) / max(gamma, 1e-8) / max(key, 1e-9))
            reservoir.push(int(idxi), 1.0, priority)
        # Controller updates
        bA = (g_cs @ v_acc_proj).mean().item()
        bC = (g_cs @ v_cal_proj).mean().item()
        prev_ema_acc, prev_ema_cal = ema_acc, ema_cal
        ema_acc = beta * ema_acc + (1 - beta) * bA
        ema_cal = beta * ema_cal + (1 - beta) * bC
        var_acc = beta * var_acc + (1 - beta) * (bA - prev_ema_acc) ** 2
        var_cal = beta * var_cal + (1 - beta) * (bC - prev_ema_cal) ** 2
        step = 0.1
        lam_acc = float(np.clip(lam_acc + step * (ema_acc - tau_acc), 0.0, 10.0))
        lam_cal = float(np.clip(lam_cal + step * (ema_cal - tau_cal), 0.0, 10.0))
        # Stability guard (increase floor when variance spikes)
        if var_acc > 10 * 1e-8 or var_cal > 10 * 1e-8:
            gamma = min(5e-3, gamma * 1.1)
        # Diagnostics
        diag["lambda_acc"].append(lam_acc)
        diag["lambda_cal"].append(lam_cal)
        diag["delta_spec"].append(delta)
        diag["spectral_tail"].append(tail)
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

    sel_time = time.time() - start
    return reservoir, sel_time, peak_mem, diag


# ------------------------------
# Baseline selectors
# ------------------------------

def select_random(N: int, budgets: Tuple[float, ...], seed: int = 13) -> Dict[float, List[int]]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    out = {}
    for b in budgets:
        k = int(max(1, b * N))
        out[b] = perm[:k].tolist()
    return out


def compute_ce_scores(X: torch.Tensor, y: torch.Tensor, head: nn.Linear) -> np.ndarray:
    with torch.no_grad():
        logits = X @ head.weight.T
        ce = F.cross_entropy(logits, y, reduction='none').cpu().numpy()
    return ce


def select_topk_by_scores(scores: np.ndarray, budgets: Tuple[float, ...]) -> Dict[float, List[int]]:
    order = np.argsort(-scores)  # descending
    N = len(scores)
    out = {}
    for b in budgets:
        k = int(max(1, b * N))
        out[b] = order[:k].tolist()
    return out


def compute_entropy_scores(X: torch.Tensor, head: nn.Linear) -> np.ndarray:
    with torch.no_grad():
        logits = X @ head.weight.T
        p = torch.softmax(logits, dim=1).cpu().numpy()
    ent = -(p * (np.log(p + 1e-12))).sum(axis=1)
    return ent


def compute_el2n_scores(X_list: List[torch.Tensor], y_list: List[torch.Tensor], head_list: List[nn.Linear]) -> np.ndarray:
    N = X_list[0].size(0)
    vals = np.zeros(N, dtype=np.float32)
    count = 0
    for X, y, head in zip(X_list, y_list, head_list):
        with torch.no_grad():
            logits = X @ head.weight.T
            ce = F.cross_entropy(logits, y, reduction='none').cpu().numpy()
        vals += ce
        count += 1
    vals /= max(1, count)
    return vals


def compute_grand_scores(X: torch.Tensor, y: torch.Tensor, head: nn.Linear) -> np.ndarray:
    with torch.no_grad():
        logits = X @ head.weight.T
        res = per_sample_head_residual(logits, y)
        g = flatten_head_grads(res, X)
        gn = g.norm(dim=1).cpu().numpy()
    return gn


# ------------------------------
# Training utilities
# ------------------------------

def train_head(
    train_loader: DataLoader,
    val_loader: DataLoader,
    head: nn.Linear,
    sample_weights: Optional[Dict[int, float]] = None,
    epochs: int = 10,
    lr: float = 2e-3,
    wd: float = 1e-2,
    seed: int = 13,
    verbose: bool = False,
) -> Tuple[List[float], float, float]:
    set_seed(seed)
    device = next(head.parameters()).device
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    losses = []
    for ep in range(epochs):
        head.train()
        for batch in train_loader:
            Xb, yb, idxb = batch[0].to(device), batch[1].to(device), batch[2]
            opt.zero_grad(set_to_none=True)
            logits = Xb @ head.weight.T
            loss = F.cross_entropy(logits, yb, reduction='none')
            if sample_weights is not None:
                w = torch.tensor([sample_weights.get(int(i), 1.0) for i in idxb], device=device)
                loss = (loss * w)
            loss = loss.mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        if verbose:
            print(f"    [train_head] Epoch {ep+1}/{epochs} loss={losses[-1]:.4f}")
    # Eval
    head.eval()
    correct, n, nll = 0, 0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            Xb, yb = batch[0].to(device), batch[1].to(device)
            logits = Xb @ head.weight.T
            nll += F.cross_entropy(logits, yb, reduction='sum').item()
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            n += yb.numel()
    acc = correct / max(1, n)
    nll = nll / max(1, n)
    return losses, acc, nll


def build_loader_from_indices(dataset: FeatureDataset, indices: List[int], batch_size: int = 256, shuffle: bool = True) -> DataLoader:
    groups = getattr(dataset, 'groups', None)
    sub_groups = groups[indices] if groups is not None else None
    sub = FeatureDataset(dataset.X[indices], dataset.y[indices], dataset.idx[indices], sub_groups)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


# ------------------------------
# End-to-end experiments (images and text)
# ------------------------------

def run_experiments(datasets: Dict[str, FeatureDataset], cfg: Dict) -> Dict:
    """Runs Experiment 1 (images) and Experiment 2 (text with fairness).

    Returns a dict of results and diagnostics for main.py to summarize/print.
    """
    images_dir = cfg["experiment"].get("images_dir", ".research/iteration1/images")
    os.makedirs(images_dir, exist_ok=True)

    device = get_device(cfg["experiment"].get("device", "auto"))
    seed = int(cfg["experiment"].get("seed", 13))
    budgets_tuple = tuple(float(b) for b in cfg["experiment"].get("budgets", [0.05, 0.1, 0.25, 0.5]))

    # Training hyperparams
    bsz = int(cfg["train"].get("batch_size", 256))
    epochs = int(cfg["train"].get("epochs", 5))
    lr = float(cfg["train"].get("lr", 5e-3))
    wd = float(cfg["train"].get("wd", 1e-2))
    head_init_std = float(cfg["model"].get("head_init_std", 0.1))

    # Selection hyperparams
    d_hat = int(cfg["selection"].get("d_hat", 128))
    r_max = int(cfg["selection"].get("r_max", 32))
    forgetting = float(cfg["selection"].get("forgetting", 0.98))
    gamma = float(cfg["selection"].get("gamma", 5e-4))
    tau_acc = float(cfg["selection"].get("tau_acc", -0.01))
    tau_cal = float(cfg["selection"].get("tau_cal", -0.005))

    set_seed(seed)

    results = {"exp1": {}, "exp2": {}}

    # --------------------
    # Experiment 1: Images
    # --------------------
    print("\n=== Experiment 1 — Image-like features (Accuracy/Calibration) ===")
    ds_tr = datasets["images_train"]
    ds_val = datasets["images_val"]
    ds_te = datasets["images_test"]
    D = ds_tr.X.size(1)
    C = int(ds_tr.y.max().item() + 1)

    train_loader = DataLoader(ds_tr, batch_size=bsz, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=bsz, shuffle=False)
    test_loader = DataLoader(ds_te, batch_size=bsz, shuffle=False)

    head = nn.Linear(D, C, bias=False).to(device)
    head.weight.data.normal_(0, head_init_std)

    # Warm-up Fisher on 10% slice
    warm_k = max(1, int(0.1 * len(ds_tr)))
    warm_idx = np.random.RandomState(seed).choice(len(ds_tr), size=warm_k, replace=False)
    warm_loader = DataLoader(FeatureDataset(ds_tr.X[warm_idx], ds_tr.y[warm_idx], ds_tr.idx[warm_idx]),
                             batch_size=bsz, shuffle=True)
    print("[Exp1] Computing diagonal Fisher on warm-up slice...")
    fisher = compute_fisher_diag(warm_loader, head)

    print("[Exp1] Computing validation gradients (acc and cal)...")
    v_acc = validation_gradient(val_loader, head, objective='acc', temperature=1.0)
    v_cal = validation_gradient(val_loader, head, objective='cal', temperature=1.5)

    P = head.weight.numel()
    s_idx, s_sign = create_s_idx_sign(P=P, d_hat=d_hat, seed=seed, device=device)

    print("[Exp1] Running OP-S&V selection pass...")
    stream_loader = DataLoader(ds_tr, batch_size=bsz, shuffle=False)
    reservoir, sel_time, peak_mem, diag = op_snv_select(
        stream_loader, head, d_hat, fisher, v_acc, v_cal, s_idx, s_sign,
        budgets=budgets_tuple, gamma=gamma, forgetting=forgetting, r_max=r_max,
        tau_acc=tau_acc, tau_cal=tau_cal, seed=seed)
    print(f"[Exp1] Selection completed in {sel_time:.3f}s; peak GPU memory: {peak_mem/1e6:.1f} MB")

    # Extract subsets per budget (nested)
    budget_indices = {}
    for b in budgets_tuple:
        k = int(max(1, b * len(ds_tr)))
        top = reservoir.topk(k)
        budget_indices[b] = [idx for idx, _ in top]
    nestedness = verify_nestedness(budget_indices)
    print(f"[Exp1] Nestedness across budgets: {nestedness*100:.1f}% (expect 100%)")

    # Baselines
    print("[Exp1] Computing baseline subsets...")
    Xtr_cpu = ds_tr.X
    ytr_cpu = ds_tr.y
    ce_scores = compute_ce_scores(Xtr_cpu, ytr_cpu, head)
    ent_scores = compute_entropy_scores(Xtr_cpu, head)
    el2n_scores = compute_el2n_scores([Xtr_cpu], [ytr_cpu], [head])
    grand_scores = compute_grand_scores(Xtr_cpu, ytr_cpu, head)

    bl_random = select_random(len(ds_tr), budgets_tuple, seed=seed)
    bl_loss = select_topk_by_scores(ce_scores, budgets_tuple)
    bl_entropy = select_topk_by_scores(ent_scores, budgets_tuple)
    bl_el2n = select_topk_by_scores(el2n_scores, budgets_tuple)
    bl_grand = select_topk_by_scores(grand_scores, budgets_tuple)

    # Train and evaluate
    res_methods = {"op-snv": {"acc": [], "nll": [], "ece": []},
                   "random": {"acc": [], "nll": [], "ece": []},
                   "loss": {"acc": [], "nll": [], "ece": []},
                   "entropy": {"acc": [], "nll": [], "ece": []},
                   "el2n": {"acc": [], "nll": [], "ece": []},
                   "grand": {"acc": [], "nll": [], "ece": []}}

    losses_last = []
    for method_name, bdict in [("op-snv", budget_indices), ("random", bl_random), ("loss", bl_loss),
                               ("entropy", bl_entropy), ("el2n", bl_el2n), ("grand", bl_grand)]:
        print(f"[Exp1] Training/evaluating method: {method_name}")
        for b in budgets_tuple:
            idxs = bdict[b]
            train_sub_loader = build_loader_from_indices(ds_tr, idxs, batch_size=bsz, shuffle=True)
            head_b = nn.Linear(D, C, bias=False).to(device)
            head_b.weight.data.copy_(head.weight.data)
            losses, acc, nll = train_head(train_sub_loader, test_loader, head_b, sample_weights=None, epochs=epochs, lr=lr, wd=wd, seed=seed)
            losses_last = losses
            # Calibration on validation using temperature scaling
            with torch.no_grad():
                logits_val = (ds_val.X.to(device) @ head_b.weight.T).cpu().numpy()
            T = eval_utils.temperature_scale_grid(logits_val, ds_val.y.numpy(), T_grid=np.linspace(0.5, 2.0, 21))
            with torch.no_grad():
                logits_test = (ds_te.X.to(device) @ head_b.weight.T).cpu().numpy()
            probs_test = torch.softmax(torch.tensor(logits_test / T), dim=1).numpy()
            ece = eval_utils.compute_ece(probs_test, ds_te.y.numpy(), n_bins=15)
            res_methods[method_name]["acc"].append(acc)
            res_methods[method_name]["nll"].append(nll)
            res_methods[method_name]["ece"].append(ece)
        # Plot training loss only for OP-S&V at the largest budget
        if method_name == "op-snv" and len(losses_last) > 0:
            eval_utils.plot_training_loss(losses_last, os.path.join(images_dir, "exp1_training_loss_op-snv.pdf"))

    # Plots for Exp1
    eval_utils.plot_metric_vs_budget(res_methods, budgets_tuple, metric="acc", out_path=os.path.join(images_dir, "exp1_accuracy_vs_budget.pdf"))
    eval_utils.plot_metric_vs_budget(res_methods, budgets_tuple, metric="ece", out_path=os.path.join(images_dir, "exp1_ece_vs_budget.pdf"))

    # Confusion matrix for OP-S&V largest budget
    if HAVE_SKLEARN:
        last_b = budgets_tuple[-1]
        idxs = budget_indices[last_b]
        head_cm = nn.Linear(D, C, bias=False).to(device)
        head_cm.weight.data.copy_(head.weight.data)
        train_sub_loader = build_loader_from_indices(ds_tr, idxs, batch_size=bsz, shuffle=True)
        _ = train_head(train_sub_loader, test_loader, head_cm, epochs=epochs, lr=lr, wd=wd, seed=seed)
        with torch.no_grad():
            logits = (ds_te.X.to(device) @ head_cm.weight.T)
            y_pred = logits.argmax(dim=1).cpu().numpy()
        cm = confusion_matrix(ds_te.y.numpy(), y_pred)
        eval_utils.plot_confusion_matrix(cm, out_path=os.path.join(images_dir, "exp1_confusion_matrix_op-snv.pdf"))

    results["exp1"] = {
        "budgets": [float(b) for b in budgets_tuple],
        "methods": res_methods,
        "nestedness": float(nestedness),
        "selection_time_sec": float(sel_time),
        "peak_mem_bytes": int(peak_mem),
    }

    # --------------------
    # Experiment 2: Text with groups (Fairness)
    # --------------------
    print("\n=== Experiment 2 — Text-like features (Fairness + Accuracy) ===")
    ds_tr = datasets["text_train"]
    ds_val = datasets["text_val"]
    ds_te = datasets["text_test"]
    D = ds_tr.X.size(1)
    C = int(ds_tr.y.max().item() + 1)
    G = int(ds_tr.groups.max().item() + 1)

    train_loader = DataLoader(ds_tr, batch_size=bsz, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=bsz, shuffle=False)
    test_loader = DataLoader(ds_te, batch_size=bsz, shuffle=False)

    head = nn.Linear(D, C, bias=False).to(device)
    head.weight.data.normal_(0, head_init_std)

    # Fisher on 20% slice
    warm_k = max(1, int(0.2 * len(ds_tr)))
    warm_idx = np.random.RandomState(seed).choice(len(ds_tr), size=warm_k, replace=False)
    warm_loader = DataLoader(FeatureDataset(ds_tr.X[warm_idx], ds_tr.y[warm_idx], ds_tr.idx[warm_idx], ds_tr.groups[warm_idx]),
                             batch_size=bsz, shuffle=True)
    print("[Exp2] Computing diagonal Fisher on warm-up slice...")
    fisher = compute_fisher_diag(warm_loader, head)

    print("[Exp2] Computing validation gradients (acc, cal, fairness surrogates)...")
    v_acc = validation_gradient(val_loader, head, objective='acc', temperature=1.0)
    v_cal = validation_gradient(val_loader, head, objective='cal', temperature=1.2)
    v_fair = eval_utils.validation_gradient_fairness(val_loader, head, G=G, alpha=5.0)

    P = head.weight.numel()
    s_idx, s_sign = create_s_idx_sign(P=P, d_hat=d_hat, seed=seed, device=device)

    # Fairness-aware selection (s_i = novelty + sum lambda_q * a_{i,q} + group floor boosts)
    def op_snv_select_fair(stream_loader: DataLoader, head: nn.Linear):
        rfd = RFD(d_hat=d_hat, r_max=r_max, forgetting=forgetting, device=device)
        P_inv = 1.0 / (fisher + 1e-6)
        def proj_vec(v):
            vP = v * P_inv
            v_cs = countsketch(vP[None, :], s_idx, s_sign, d_hat)[0]
            U = rfd.basis()
            return U @ (U.T @ v_cs)
        vA = proj_vec(v_acc); vC = proj_vec(v_cal); vF = proj_vec(v_fair)
        nA = vA.norm() + 1e-8; nC = vC.norm() + 1e-8; nF = vF.norm() + 1e-8
        lamA, lamC, lamF = 1.0, 1.0, 1.0
        emaA, emaC, emaF = 0.0, 0.0, 0.0
        budgets_local = tuple(float(b) for b in cfg["experiment"].get("budgets_fair", [0.05, 0.1, 0.25]))
        reservoir = NestedReservoir(max_k=int(max(budgets_local)*len(stream_loader.dataset)), seed=seed)
        gamma_local, beta = max(1e-4, gamma), 0.9
        for batch in stream_loader:
            Xb, yb, idxb, gb = batch[0].to(device), batch[1].to(device), batch[2], batch[3].to(device)
            logits = Xb @ head.weight.T
            res = per_sample_head_residual(logits, yb)
            g = flatten_head_grads(res, Xb)
            g_cs = countsketch(g, s_idx, s_sign, d_hat)
            tail, delta = rfd.update(g_cs)
            U = rfd.basis()
            proj = g_cs @ U @ U.T
            novelty = (g_cs - proj).pow(2).sum(dim=1)
            aA = torch.clamp(-(g_cs @ vA) / nA, min=0)
            aC = torch.clamp(-(g_cs @ vC) / nC, min=0)
            aF = torch.clamp(-(g_cs @ vF) / nF, min=0)
            # Per-group probability floors via score boost
            group_counts = torch.bincount(gb, minlength=G).float()
            prop = group_counts / max(1, gb.numel())
            floor_boost = torch.tensor([float(0.01/(prop[g].item()+1e-6)) for g in gb], device=device)
            score = novelty + lamA*aA + lamC*aC + lamF*aF + floor_boost
            for s, idxi in zip(score, idxb):
                key = (hash(int(idxi)) & 0xffffffff) / 2**32
                priority = float((s.item() + 1e-8) / max(gamma_local, 1e-8) / max(key, 1e-9))
                reservoir.push(int(idxi), 1.0, priority)
            # Controller with simple EMA residuals
            bA = (g_cs @ vA).mean().item(); bC = (g_cs @ vC).mean().item(); bF = (g_cs @ vF).mean().item()
            emaA = beta*emaA + (1-beta)*bA
            emaC = beta*emaC + (1-beta)*bC
            emaF = beta*emaF + (1-beta)*bF
            lamA = float(np.clip(lamA + 0.1*(emaA + 0.01), 0.0, 10.0))
            lamC = float(np.clip(lamC + 0.1*(emaC + 0.005), 0.0, 10.0))
            lamF = float(np.clip(lamF + 0.1*(emaF + 0.005), 0.0, 10.0))
        return reservoir, budgets_local

    print("[Exp2] Running OP-S&V fairness-aware selection pass...")
    stream_loader = DataLoader(ds_tr, batch_size=bsz, shuffle=False)
    reservoir_fair, budgets_fair = op_snv_select_fair(stream_loader, head)

    budget_indices = {}
    for b in budgets_fair:
        k = int(max(1, b * len(ds_tr)))
        budget_indices[b] = [idx for idx, _ in reservoir_fair.topk(k)]

    # Baseline: Random stratified by group proportion
    rng = np.random.RandomState(seed)
    bl_random = {}
    for b in budgets_fair:
        k = int(max(1, b * len(ds_tr)))
        idxs = []
        for g in range(G):
            g_idx = np.where(ds_tr.groups.numpy() == g)[0]
            take = max(0, int(round(len(g_idx) / len(ds_tr) * k)))
            if take > 0:
                sel = rng.choice(g_idx, size=min(take, len(g_idx)), replace=False)
                idxs.extend(sel.tolist())
        if len(idxs) < k:
            remain = list(set(range(len(ds_tr))) - set(idxs))
            fill = rng.choice(remain, size=k - len(idxs), replace=False).tolist()
            idxs.extend(fill)
        bl_random[b] = idxs[:k]

    # Train/Eval: overall accuracy and worst-group accuracy
    res_methods_fair = {"op-snv": {"acc": [], "worst_group_acc": []},
                        "random": {"acc": [], "worst_group_acc": []}}

    def evaluate_worst_group_acc(head_model: nn.Linear) -> Tuple[float, float]:
        head_model.eval()
        with torch.no_grad():
            logits = (ds_te.X.to(device) @ head_model.weight.T)
            pred = logits.argmax(dim=1).cpu()
        overall = (pred.numpy() == ds_te.y.numpy()).mean()
        wg = 1.0
        for g in range(G):
            m = (ds_te.groups.numpy() == g)
            if m.sum() > 0:
                accg = (pred.numpy()[m] == ds_te.y.numpy()[m]).mean()
                wg = min(wg, accg)
        return overall, wg

    for name, bdict in [("op-snv", budget_indices), ("random", bl_random)]:
        print(f"[Exp2] Training/evaluating method: {name}")
        for b in budgets_fair:
            idxs = bdict[b]
            loader = build_loader_from_indices(ds_tr, idxs, batch_size=bsz, shuffle=True)
            head_b = nn.Linear(D, C, bias=False).to(device)
            head_b.weight.data.copy_(head.weight.data)
            _ = train_head(loader, test_loader, head_b, epochs=epochs, lr=lr, wd=wd, seed=seed)
            acc, wg = evaluate_worst_group_acc(head_b)
            res_methods_fair[name]["acc"].append(acc)
            res_methods_fair[name]["worst_group_acc"].append(wg)

    # Pareto plot: worst-group vs overall
    eval_utils.plot_pareto_fairness(res_methods_fair, out_path=os.path.join(images_dir, "exp2_pareto_fairness.pdf"))

    results["exp2"] = {
        "budgets": [float(b) for b in budgets_fair],
        "methods": res_methods_fair,
    }

    # Save diagnostics
    diag_path = os.path.join(images_dir, "diagnostics_summary.json")
    try:
        with open(diag_path, "w") as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass

    return results
