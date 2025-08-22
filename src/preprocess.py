"""
FLARE-MoE Data Preprocessing

This module implements data preprocessing utilities for FLARE-MoE experiments:
- Synthetic batch generation for testing
- Network simulation utilities
- Byte estimation functions
"""

import math
import time
import random
from typing import List, Tuple, Dict

import numpy as np
import torch


class TrueNetwork:
    """Underlying (unknown to router) network model for elapsed time simulation."""
    def __init__(self, world_size: int, intra_beta: float = 2e-9, inter_beta: float = 8e-9, 
                 alpha_base: float = 3e-5, inter_extra_alpha: float = 2e-4, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.world_size = world_size
        self.alpha = alpha_base * (1 + 0.1 * torch.randn(world_size, world_size, generator=g))
        self.beta = torch.full((world_size, world_size), intra_beta)
        
        for i in range(world_size):
            for j in range(world_size):
                if i != j and ((i + j) % 3 == 0):
                    self.beta[i, j] = inter_beta
                    self.alpha[i, j] += inter_extra_alpha

    def elapsed(self, src: int, dst: int, bytes_sent: float, contention_scale: float = 1.0) -> float:
        a = float(self.alpha[src, dst].item())
        b = float(self.beta[src, dst].item()) * contention_scale
        noise = max(0.0, np.random.normal(0, 0.00002))
        return a + b * bytes_sent + noise


def generate_synthetic_batch(batch_T: int, ctx_dim: int, vocab_size: int, seq_len: int, 
                           seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic batch data for testing."""
    g = torch.Generator().manual_seed(seed)
    x_t = torch.randn(batch_T, ctx_dim, generator=g)
    token_ids = torch.randint(low=0, high=vocab_size, size=(batch_T,), generator=g)
    pos_ids = torch.randint(low=0, high=seq_len, size=(batch_T,), generator=g)
    return x_t, token_ids, pos_ids


def estimate_bytes_per_dst_from_uniform(T: int, k: int, E: int, experts_per_rank: int, 
                                      element_size: int, hidden_dim: int, world_size: int) -> torch.Tensor:
    """Estimate bytes per destination assuming uniform expert selection."""
    tokens_per_expert = (T * k) / E
    bytes_per_expert = tokens_per_expert * hidden_dim * element_size
    bytes_per_dst = torch.zeros(world_size, dtype=torch.float32)
    for e in range(E):
        dst = e // experts_per_rank
        bytes_per_dst[dst] += bytes_per_expert
    return bytes_per_dst


def tier_of_assignment(device_id: int, expert: int, experts_per_rank: int, rank2node: List[int]) -> str:
    """Determine which tier an expert assignment belongs to."""
    dst_rank = expert // experts_per_rank
    if dst_rank == device_id:
        return "local"
    if rank2node[dst_rank] == rank2node[device_id]:
        return "intra"
    return "inter"


def estimate_flops_per_token(selected_experts_per_token: List[List[int]], d_model: int, ffn_dim: int) -> float:
    """Estimate FLOPs per token from expert selections."""
    flops_expert = 2.0 * d_model * ffn_dim
    total = 0.0
    for sel in selected_experts_per_token:
        total += flops_expert * len(sel)
    return total / max(1, len(selected_experts_per_token))


def create_contention_schedule(num_steps: int, base_scale: float = 1.0, 
                             peak_scale: float = 3.0, seed: int = 0) -> List[float]:
    """Create a dynamic contention schedule for network simulation."""
    np.random.seed(seed)
    schedule = []
    for step in range(num_steps):
        phase = (step / num_steps) * 2 * np.pi
        base_contention = base_scale + 0.5 * (peak_scale - base_scale) * (1 + np.sin(phase))
        noise = np.random.normal(0, 0.1 * base_contention)
        schedule.append(max(0.1, base_contention + noise))
    return schedule
