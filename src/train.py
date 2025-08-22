"""
FLARE-MoE Training Components

This module implements the core FLARE-MoE router and training utilities:
- BudgetedHierarchicalTopKRouter: Main router with all FLARE features
- CostCache: Live alpha-beta cost tracking
- PIController: Closed-loop budget control
- Supporting utilities for hierarchical routing and fairness
"""

import math
import time
import random
import socket
import hashlib
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CostCache:
    """Live alpha-beta cost cache C with EMA updates.
    c_ij = alpha_ij + beta_ij * bytes
    """
    def __init__(self, world_size: int, ema: float = 0.1, device: str = "cpu"):
        self.world_size = world_size
        self.device = device
        self.alpha = torch.zeros(world_size, world_size, device=device)
        self.beta = torch.ones(world_size, world_size, device=device) * 1e-9
        self.ema = ema

    @torch.no_grad()
    def update(self, src: int, dst: int, bytes_sent: float, elapsed_s: float):
        if bytes_sent <= 0 or elapsed_s <= 0:
            return
        beta_hat = max(elapsed_s / bytes_sent, 0.0)
        alpha_hat = max(elapsed_s - beta_hat * bytes_sent, 0.0)
        self.beta[src, dst] = (1 - self.ema) * self.beta[src, dst] + self.ema * beta_hat
        self.alpha[src, dst] = (1 - self.ema) * self.alpha[src, dst] + self.ema * alpha_hat

    @torch.no_grad()
    def normed_costs(self, src: int, bytes_est_per_dst: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        costs = self.alpha[src] + self.beta[src] * bytes_est_per_dst.to(self.device)
        m, v = costs.mean(), costs.var(unbiased=False)
        return (costs - m) / torch.sqrt(v + eps)


class PIController:
    """PI Controller for closed-loop budget control."""
    def __init__(self, kp: float = 0.05, ki: float = 0.001, u_min: float = 0.0, u_max: float = 10.0, anti_windup: float = 10.0):
        self.kp = kp
        self.ki = ki
        self.u_min = u_min
        self.u_max = u_max
        self.anti_windup = anti_windup
        self.integral = 0.0
        self.u = 0.0

    def step(self, error: float) -> float:
        self.integral = float(max(min(self.integral + error, self.anti_windup), -self.anti_windup))
        self.u = float(max(min(self.kp * error + self.ki * self.integral, self.u_max), self.u_min))
        return self.u


class GradientReversalFn(torch.autograd.Function):
    """Gradient reversal for adversarial debiasing."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return GradientReversalFn.apply(x, lambd)


def get_node_id() -> int:
    """Get node ID from hostname for topology awareness."""
    host = socket.gethostname().encode("utf-8")
    return int(hashlib.md5(host).hexdigest()[:8], 16)


def build_tiers_for_rank(rank: int, world_size: int, experts_per_rank: int, rank2node: List[int]) -> List[List[int]]:
    """Build hierarchical tiers: local -> intra-node -> inter-node."""
    E = world_size * experts_per_rank
    expert_of_rank = {}
    e_start = 0
    for r in range(world_size):
        expert_of_rank[r] = list(range(e_start, e_start + experts_per_rank))
        e_start += experts_per_rank
    
    local = expert_of_rank[rank]
    intra = []
    inter = []
    for r in range(world_size):
        if r == rank:
            continue
        if rank2node[r] == rank2node[rank]:
            intra.extend(expert_of_rank[r])
        else:
            inter.extend(expert_of_rank[r])
    return [local, intra, inter]


def make_device_map(world_size: int, experts_per_rank: int) -> List[int]:
    """Create mapping from expert ID to device ID."""
    E = world_size * experts_per_rank
    device_map = []
    for e in range(E):
        device_map.append(e // experts_per_rank)
    return device_map


class BudgetedHierarchicalTopKRouter(nn.Module):
    """
    FLARE-MoE Router: Budgeted Hierarchical Top-K with fairness and debiasing.
    
    Features:
    1. Budgeted hierarchical routing with live cost tracking
    2. Position-aware fairness to prevent late-token starvation
    3. Token-ID debiasing via router factorization
    4. Compute-aware expert skipping for inference
    """
    
    def __init__(self,
                 num_experts: int,
                 k: int,
                 tiers: List[List[int]],
                 device_map: List[int],
                 cost_cache: CostCache,
                 bytes_budget: Optional[float] = None,
                 flops_budget: Optional[float] = None,
                 delta_margin: float = 0.5,
                 mu_gains: Tuple[float, float] = (0.05, 0.001),
                 tau_gains: Tuple[float, float] = (0.05, 0.001),
                 mu_max: float = 10.0,
                 lambda_pos: float = 0.0,
                 num_windows: int = 8,
                 window_caps: Optional[float] = None,
                 token_vocab_size: int = 10000,
                 ctx_dim: int = 128,
                 adv_debias_weight: float = 0.0,
                 adv_lambda: float = 1.0,
                 distill_weight: float = 0.0,
                 device_id: int = 0,
                 device: str = "cpu"):
        super().__init__()
        self.E = num_experts
        self.k = k
        self.tiers = [list(t) for t in tiers]
        self.device_map = device_map
        self.C = cost_cache
        self.bytes_budget = bytes_budget
        self.flops_budget = flops_budget
        self.delta_margin = delta_margin
        self.mu_ctl = PIController(mu_gains[0], mu_gains[1], 0.0, mu_max)
        self.tau_ctl = PIController(tau_gains[0], tau_gains[1], 0.0, 20.0)
        self.mu = 0.0
        self.tau = 0.0
        self.lambda_pos = lambda_pos
        self.W = num_windows
        self.window_caps = window_caps
        self.device_id = device_id
        self.device = device
        
        self.id_bias = nn.Embedding(token_vocab_size, self.E)
        self.ctx_head = nn.Linear(ctx_dim, ctx_dim, bias=False)
        self.U = nn.Parameter(torch.empty(ctx_dim, self.E))
        nn.init.normal_(self.U, mean=0.0, std=0.02)
        
        self.adv_head = nn.Linear(self.E, self.E)
        self.adv_debias_weight = adv_debias_weight
        self.adv_lambda = adv_lambda
        self.distill_weight = distill_weight
        
        self.register_buffer("used_e_w", torch.zeros(self.E, self.W))

    def _window_index(self, pos_ids: torch.Tensor) -> torch.Tensor:
        """Map position IDs to window indices for fairness tracking."""
        max_pos = max(int(pos_ids.max().item()), 1)
        return torch.clamp((pos_ids * self.W) // (max_pos + 1), 0, self.W - 1)

    def _map_dst_costs_to_experts(self, costs_norm_dst: torch.Tensor) -> torch.Tensor:
        """Map per-destination costs to per-expert costs."""
        idx = torch.tensor(self.device_map, device=costs_norm_dst.device, dtype=torch.long)
        return costs_norm_dst[idx]

    def forward(self,
                x_t: torch.Tensor,
                token_ids: torch.Tensor,
                pos_ids: torch.Tensor,
                bytes_est_per_dst: torch.Tensor,
                training: bool = True,
                compute_skip: bool = False) -> Tuple[List[List[int]], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of FLARE-MoE router.
        
        Returns:
            assignments: List of expert assignments per token
            logits: Router logits (for analysis)
            losses: Dictionary of auxiliary losses
        """
        T, D = x_t.shape
        E = self.E
        
        h = self.ctx_head(x_t)
        s_teacher = h @ self.U + self.id_bias(token_ids)
        s = s_teacher.clone()
        
        if self.C is not None and bytes_est_per_dst is not None:
            costs_norm_dst = self.C.normed_costs(self.device_id, bytes_est_per_dst).to(s.device)
            costs_e = self._map_dst_costs_to_experts(costs_norm_dst)
            s = s - self.mu * costs_e[None, :]
        
        if self.lambda_pos > 0.0:
            w = self._window_index(pos_ids)
            used = self.used_e_w[:, w].T  # [T, E]
            if self.window_caps is None or math.isinf(self.window_caps):
                caps = torch.full_like(used, float("inf"))
            else:
                caps = torch.full_like(used, float(self.window_caps))
            over = torch.relu(used - caps)
            s = s - self.lambda_pos * over
        
        skip_mask = torch.zeros(T, dtype=torch.bool, device=s.device)
        k_eff = torch.full((T,), self.k, dtype=torch.long, device=s.device)
        if compute_skip and not training:
            top2 = torch.topk(s, k=min(2, E), dim=-1).values
            if top2.size(1) == 2:
                margin = top2[:, 0] - top2[:, 1]
            else:
                margin = top2[:, 0]
            skip_mask = margin < self.tau
            k_eff = torch.where(margin > (2 * self.tau), torch.ones_like(k_eff), torch.full_like(k_eff, self.k))
        
        assign = []
        for t in range(T):
            if skip_mask[t]:
                assign.append([])
                continue
            
            need = int(k_eff[t].item())
            selected: List[int] = []
            global_best = int(torch.argmax(s[t]).item())
            
            for tier in self.tiers:
                if need <= 0:
                    break
                
                tier_idx = torch.tensor(tier, device=s.device, dtype=torch.long)
                if len(tier_idx) == 0:
                    continue
                    
                tier_scores = s[t, tier_idx]
                kk = min(need, tier_scores.numel())
                vals, idxs = torch.topk(tier_scores, k=kk)
                selected.extend(tier_idx[idxs].tolist())
                need -= kk
                
                if need > 0 and (float(s[t, global_best].item()) - float(vals.max().item())) > self.delta_margin:
                    continue
                if need <= 0:
                    break
            
            if len(selected) == 0:
                selected = [global_best]
            assign.append(selected[:int(k_eff[t].item())])
        
        if self.lambda_pos > 0.0 and training:
            with torch.no_grad():
                w = self._window_index(pos_ids)
                for t, sel in enumerate(assign):
                    for e in sel:
                        self.used_e_w[e, int(w[t].item())] += 1
        
        loss_adv = torch.tensor(0.0, device=s.device)
        loss_kld = torch.tensor(0.0, device=s.device)
        
        return assign, s, {"loss_adv": loss_adv, "loss_kld": loss_kld}

    def update_mu(self, bytes_obs: float) -> float:
        """Update μ parameter via PI controller for bytes budget."""
        if self.bytes_budget is None:
            return self.mu
        err = float(bytes_obs - self.bytes_budget)
        self.mu = self.mu_ctl.step(err)
        return self.mu

    def update_tau(self, flops_obs: float) -> float:
        """Update τ parameter via PI controller for FLOPs budget."""
        if self.flops_budget is None:
            return self.tau
        err = float(flops_obs - self.flops_budget)
        self.tau = self.tau_ctl.step(err)
        return self.tau

    def reset_fairness_counters(self):
        """Reset fairness counters (e.g., between sequences)."""
        self.used_e_w.zero_()
