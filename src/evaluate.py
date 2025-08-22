# Evaluation and auditing utilities (src/evaluate.py)
# - Model evaluation
# - Transparency ledger reader
# - Auditor's reconstruction of DDG-TREE aggregate noise

import os
import json
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

from typing import Dict, Any


def evaluate_model(model: torch.nn.Module, test_loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.numel())
    return 100.0 * correct / max(total, 1)


def write_ledger_record(path: str, record_dict: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a") as f:
        f.write(json.dumps(record_dict) + "\n")


def read_ledger(path: str):
    with open(path, "r") as f:
        return [json.loads(l) for l in f]


# Auditor-side copy of DDG integer Gaussian

def _ddg_int_gaussian(shape, sigma, rng: np.random.Generator):
    return np.rint(rng.normal(0.0, sigma, size=shape)).astype(np.int64)


def auditor_reconstruct_noise(round_t: int, sketch_dim: int, seeds: dict, target_var_per_node: dict, L_min: int, realized_L: int):
    rngs = {k: np.random.default_rng(seeds[k]) for k in seeds}
    agg = np.zeros(sketch_dim, dtype=np.int64)
    # Reconstruct per DP-FTRL tree nodes used up to round_t
    def _tree_nodes_for_round(t: int):
        nodes = []
        k, x = 0, t
        while x > 0:
            if x & 1:
                nodes.append((k, (t >> k) << k))
            k += 1
            x >>= 1
        return nodes
    for node in _tree_nodes_for_round(round_t):
        if node not in target_var_per_node:
            continue
        sigma = np.sqrt(target_var_per_node[node])
        # Aggregate level (post-SecAgg) noise uses sigma (client shares sum)
        z = _ddg_int_gaussian(sketch_dim, sigma, rngs[node])
        agg += z
    return agg
