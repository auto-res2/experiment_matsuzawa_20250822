#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing and synthetic data generation for OP-S&V experiments.
This module produces feature-level datasets to emulate frozen encoder outputs
for both image-like and text-like tasks.
"""

from typing import Dict, Optional
import os
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, idx: torch.Tensor,
                 groups: Optional[torch.Tensor] = None):
        self.X = X
        self.y = y
        self.idx = idx
        self.groups = groups
    def __len__(self):
        return self.X.size(0)
    def __getitem__(self, i):
        if self.groups is None:
            return self.X[i], self.y[i], self.idx[i]
        else:
            return self.X[i], self.y[i], self.idx[i], self.groups[i]


def make_image_synthetic(cfg: Dict) -> Dict[str, FeatureDataset]:
    N_train = int(cfg["data"]["image"].get("N_train", 1500))
    N_val = int(cfg["data"]["image"].get("N_val", 300))
    N_test = int(cfg["data"]["image"].get("N_test", 300))
    D = int(cfg["data"]["image"].get("D", 64))
    C = int(cfg["data"]["image"].get("C", 5))

    X_train = torch.randn(N_train, D)
    X_val = torch.randn(N_val, D)
    X_test = torch.randn(N_test, D)
    W_true = torch.randn(C, D)
    y_train = (X_train @ W_true.T + 0.5 * torch.randn(N_train, C)).argmax(dim=1)
    y_val = (X_val @ W_true.T + 0.5 * torch.randn(N_val, C)).argmax(dim=1)
    y_test = (X_test @ W_true.T + 0.5 * torch.randn(N_test, C)).argmax(dim=1)

    idx_train = torch.arange(N_train)
    idx_val = torch.arange(N_val)
    idx_test = torch.arange(N_test)

    ds_train = FeatureDataset(X_train, y_train, idx_train)
    ds_val = FeatureDataset(X_val, y_val, idx_val)
    ds_test = FeatureDataset(X_test, y_test, idx_test)

    return {"images_train": ds_train, "images_val": ds_val, "images_test": ds_test}


def make_text_synthetic(cfg: Dict) -> Dict[str, FeatureDataset]:
    N_train = int(cfg["data"]["text"].get("N_train", 1200))
    N_val = int(cfg["data"]["text"].get("N_val", 300))
    N_test = int(cfg["data"]["text"].get("N_test", 300))
    D = int(cfg["data"]["text"].get("D", 48))
    C = int(cfg["data"]["text"].get("C", 2))
    G = int(cfg["data"]["text"].get("G", 4))

    group_means = torch.randn(G, D) * 0.5
    groups_train = torch.randint(0, G, (N_train,))
    groups_val = torch.randint(0, G, (N_val,))
    groups_test = torch.randint(0, G, (N_test,))

    X_train = torch.randn(N_train, D) + group_means[groups_train]
    X_val = torch.randn(N_val, D) + group_means[groups_val]
    X_test = torch.randn(N_test, D) + group_means[groups_test]
    w_true = torch.randn(C, D)
    y_train = (X_train @ w_true.T + 0.3 * torch.randn(N_train, C)).argmax(dim=1)
    y_val = (X_val @ w_true.T + 0.3 * torch.randn(N_val, C)).argmax(dim=1)
    y_test = (X_test @ w_true.T + 0.3 * torch.randn(N_test, C)).argmax(dim=1)

    idx_train = torch.arange(N_train)
    idx_val = torch.arange(N_val)
    idx_test = torch.arange(N_test)

    ds_train = FeatureDataset(X_train, y_train, idx_train, groups_train)
    ds_val = FeatureDataset(X_val, y_val, idx_val, groups_val)
    ds_test = FeatureDataset(X_test, y_test, idx_test, groups_test)

    return {"text_train": ds_train, "text_val": ds_val, "text_test": ds_test}


def preprocess(cfg: Dict) -> Dict[str, FeatureDataset]:
    os.makedirs(cfg.get("data_dir", "data"), exist_ok=True)
    # Generate synthetic datasets for quick functionality tests and figures
    out = {}
    out.update(make_image_synthetic(cfg))
    out.update(make_text_synthetic(cfg))
    return out
