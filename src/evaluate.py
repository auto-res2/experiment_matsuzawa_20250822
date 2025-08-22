#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation utilities and plotting for OP-S&V experiments.
All figures are saved as high-quality PDF files suitable for academic papers.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def temperature_scale_grid(logits: np.ndarray, y: np.ndarray, T_grid: np.ndarray) -> float:
    best_T, best_nll = 1.0, float('inf')
    for T in T_grid:
        p = torch.softmax(torch.tensor(logits / T), dim=1).numpy()
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T


def compute_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confidences > bins[i]) & (confidences <= bins[i+1])
        if m.sum() > 0:
            acc = accuracies[m].mean()
            conf = confidences[m].mean()
            ece += (m.mean()) * abs(acc - conf)
    return float(ece)


def plot_metric_vs_budget(results: Dict[str, Dict[str, List[float]]], budgets: Tuple[float, ...], metric: str, out_path: str):
    plt.figure(figsize=(5.2, 3.3))
    palette = sns.color_palette(n_colors=len(results))
    for (mname, mres), col in zip(results.items(), palette):
        xs = [int(100*b) for b in budgets]
        ys = mres[metric]
        plt.plot(xs, ys, marker='o', label=mname, color=col, linewidth=1.6, markersize=4.0)
    plt.xlabel("Budget (%)")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs budget")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()


def plot_training_loss(losses: List[float], out_path: str):
    plt.figure(figsize=(5.2, 3.3))
    plt.plot(list(range(len(losses))), losses, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.title("Training loss (OP-S&V, last budget)")
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: str):
    plt.figure(figsize=(4.2, 4.0))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (OP-S&V)")
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()


def plot_pareto_fairness(results_fair: Dict[str, Dict[str, List[float]]], out_path: str):
    plt.figure(figsize=(5.2, 3.6))
    palette = sns.color_palette(n_colors=len(results_fair))
    for (name, vals), col in zip(results_fair.items(), palette):
        xs = vals["acc"]
        ys = vals["worst_group_acc"]
        plt.plot(xs, ys, marker='o', label=name, color=col, linewidth=1.6, markersize=4.0)
    plt.xlabel("Overall accuracy")
    plt.ylabel("Worst-group accuracy")
    plt.title("Pareto: fairness vs accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()


def fairness_loss_pairwise(logits: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, G: int, alpha: float = 5.0) -> torch.Tensor:
    # Approximate equalized odds via TPR/FPR gaps averaged over groups
    y_pos = (y == 1)
    y_neg = (y == 0)
    def soft_rate(z: torch.Tensor, positive: bool = True) -> torch.Tensor:
        if z.numel() == 0:
            return logits.new_tensor(0.5)
        s = torch.sigmoid(alpha * z)
        return s.mean() if positive else (1 - s).mean()
    # Reduce multi-class logits to binary score using class-1 vs class-0 margin if needed
    if logits.dim() == 2:
        z_all = logits[:, 1] - logits[:, 0]
    else:
        z_all = logits
    tpr_all = soft_rate(z_all[y_pos], positive=True)
    fpr_all = soft_rate(z_all[y_neg], positive=False)
    loss = logits.new_tensor(0.0)
    for g in range(G):
        mg = (groups == g)
        tpr_g = soft_rate(z_all[mg & y_pos], positive=True)
        fpr_g = soft_rate(z_all[mg & y_neg], positive=False)
        loss = loss + (tpr_g - tpr_all).abs() + (fpr_g - fpr_all).abs()
    return loss / max(1, G)


def validation_gradient_fairness(val_loader, head: torch.nn.Linear, G: int, alpha: float = 5.0) -> torch.Tensor:
    device = next(head.parameters()).device
    head.zero_grad(set_to_none=True)
    total_loss = 0.0
    for batch in val_loader:
        if len(batch) != 4:
            raise ValueError("Fairness validation loader must provide (X, y, idx, group)")
        Xb, yb, _, gb = batch
        Xb = Xb.to(device); yb = yb.to(device); gb = gb.to(device)
        logits = (Xb @ head.weight.T)
        loss = fairness_loss_pairwise(logits, yb, gb, G, alpha=alpha)
        total_loss = total_loss + loss
    total_loss.backward()
    v = head.weight.grad.view(-1).detach()
    head.zero_grad(set_to_none=True)
    return v
