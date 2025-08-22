"""
FLARE-MoE Evaluation Functions

This module implements the three main experiments for FLARE-MoE:
1. Communication budget control with μ-controller
2. Position-aware fairness (anti-starvation)
3. Compute-aware expert skipping
"""

import math
import time
import random
import os
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from train import BudgetedHierarchicalTopKRouter, CostCache, PIController, build_tiers_for_rank, make_device_map
from preprocess import generate_synthetic_batch, estimate_bytes_per_dst_from_uniform, TrueNetwork, create_contention_schedule, tier_of_assignment, estimate_flops_per_token


def run_exp1_comm_budget(seed: int = 0,
                        world_size: int = 4,
                        experts_per_rank: int = 2,
                        batch_T: int = 128,
                        ctx_dim: int = 64,
                        vocab_size: int = 1000,
                        seq_len: int = 512,
                        bytes_budget: float = 50000.0,
                        num_steps: int = 50,
                        output_dir: str = ".") -> Dict[str, float]:
    """
    Experiment 1: Communication Budget Control with μ-Controller
    
    Tests the closed-loop budget control system under dynamic network conditions.
    """
    print(f"🔧 Exp1: Communication budget control (target: {bytes_budget:.0f} bytes)")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    E = world_size * experts_per_rank
    device_map = make_device_map(world_size, experts_per_rank)
    rank2node = [i // 2 for i in range(world_size)]
    tiers = build_tiers_for_rank(0, world_size, experts_per_rank, rank2node)
    
    cost_cache = CostCache(world_size, ema=0.1, device="cpu")
    true_network = TrueNetwork(world_size, seed=seed)
    contention_schedule = create_contention_schedule(num_steps, base_scale=1.0, peak_scale=3.0, seed=seed)
    
    router = BudgetedHierarchicalTopKRouter(
        num_experts=E,
        k=2,
        tiers=tiers,
        device_map=device_map,
        cost_cache=cost_cache,
        bytes_budget=bytes_budget,
        mu_gains=(0.05, 0.001),
        token_vocab_size=vocab_size,
        ctx_dim=ctx_dim,
        device_id=0,
        device="cpu"
    )
    
    mu_history = []
    bytes_history = []
    tier_shares = {"local": [], "intra": [], "inter": []}
    
    for step in range(num_steps):
        x_t, token_ids, pos_ids = generate_synthetic_batch(batch_T, ctx_dim, vocab_size, seq_len, seed + step)
        bytes_est = estimate_bytes_per_dst_from_uniform(batch_T, 2, E, experts_per_rank, 4, ctx_dim, world_size)
        
        assignments, logits, losses = router(x_t, token_ids, pos_ids, bytes_est, training=True)
        
        total_bytes = 0.0
        tier_counts = {"local": 0, "intra": 0, "inter": 0}
        
        for t, selected in enumerate(assignments):
            for expert in selected:
                dst_rank = device_map[expert]
                bytes_sent = ctx_dim * 4
                total_bytes += bytes_sent
                
                tier = tier_of_assignment(0, expert, experts_per_rank, rank2node)
                tier_counts[tier] += 1
                
                contention = contention_schedule[step]
                elapsed = true_network.elapsed(0, dst_rank, bytes_sent, contention)
                cost_cache.update(0, dst_rank, bytes_sent, elapsed)
        
        avg_bytes_per_token = total_bytes / batch_T
        bytes_history.append(avg_bytes_per_token)
        
        total_assignments = sum(tier_counts.values())
        for tier in tier_shares:
            share = tier_counts[tier] / max(1, total_assignments)
            tier_shares[tier].append(share)
        
        mu = router.update_mu(total_bytes)
        mu_history.append(mu)
        
        if step % 10 == 0:
            print(f"   Step {step:2d}: μ={mu:.3f}, bytes/token={avg_bytes_per_token:.1f}, "
                  f"local={tier_shares['local'][-1]:.2f}")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    steps = range(num_steps)
    ax1.plot(steps, mu_history, 'b-', linewidth=2, label='μ (control signal)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('μ value')
    ax1.set_title('Communication Budget Controller')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(steps, bytes_history, 'r-', linewidth=2, label='Observed bytes/token')
    ax2.axhline(y=bytes_budget/batch_T, color='g', linestyle='--', linewidth=2, label='Target')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Bytes per token')
    ax2.set_title('Bytes Budget Tracking')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    for tier, color in [('local', 'green'), ('intra', 'orange'), ('inter', 'red')]:
        ax3.plot(steps, tier_shares[tier], color=color, linewidth=2, label=f'{tier.capitalize()} tier')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Tier share')
    ax3.set_title('Hierarchical Routing Shares')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4.plot(steps, contention_schedule, 'purple', linewidth=2, label='Network contention')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Contention scale')
    ax4.set_title('Dynamic Network Conditions')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_comm_budget.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    final_mu = mu_history[-1]
    avg_bytes_per_token = np.mean(bytes_history[-10:])
    target_bytes_per_token = bytes_budget / batch_T
    budget_adherence = 1.0 - abs(avg_bytes_per_token - target_bytes_per_token) / target_bytes_per_token
    
    print(f"   ✅ Final μ: {final_mu:.4f}")
    print(f"   📊 Avg bytes/token: {avg_bytes_per_token:.2f} (target: {target_bytes_per_token:.2f})")
    print(f"   🎯 Budget adherence: {budget_adherence:.1%}")
    
    return {
        "final_mu": final_mu,
        "avg_bytes_per_token": avg_bytes_per_token,
        "budget_adherence": budget_adherence
    }


def run_exp2_position_fairness(seed: int = 0,
                              world_size: int = 4,
                              experts_per_rank: int = 2,
                              batch_T: int = 256,
                              ctx_dim: int = 64,
                              vocab_size: int = 1000,
                              seq_len: int = 1024,
                              lambda_pos: float = 0.5,
                              num_windows: int = 8,
                              window_caps: float = 20.0,
                              num_steps: int = 100,
                              output_dir: str = ".") -> Dict[str, float]:
    """
    Experiment 2: Position-Aware Fairness (Anti-Starvation)
    
    Tests the position fairness mechanism to prevent late-token starvation.
    """
    print(f"🔧 Exp2: Position fairness (λ_pos={lambda_pos}, windows={num_windows})")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    E = world_size * experts_per_rank
    device_map = make_device_map(world_size, experts_per_rank)
    rank2node = [i // 2 for i in range(world_size)]
    tiers = build_tiers_for_rank(0, world_size, experts_per_rank, rank2node)
    
    cost_cache = CostCache(world_size, ema=0.1, device="cpu")
    
    router_fair = BudgetedHierarchicalTopKRouter(
        num_experts=E,
        k=2,
        tiers=tiers,
        device_map=device_map,
        cost_cache=cost_cache,
        lambda_pos=lambda_pos,
        num_windows=num_windows,
        window_caps=window_caps,
        token_vocab_size=vocab_size,
        ctx_dim=ctx_dim,
        device_id=0,
        device="cpu"
    )
    
    router_unfair = BudgetedHierarchicalTopKRouter(
        num_experts=E,
        k=2,
        tiers=tiers,
        device_map=device_map,
        cost_cache=cost_cache,
        lambda_pos=0.0,
        num_windows=num_windows,
        window_caps=window_caps,
        token_vocab_size=vocab_size,
        ctx_dim=ctx_dim,
        device_id=0,
        device="cpu"
    )
    
    position_utilization_fair = defaultdict(list)
    position_utilization_unfair = defaultdict(list)
    
    for step in range(num_steps):
        x_t, token_ids, pos_ids = generate_synthetic_batch(batch_T, ctx_dim, vocab_size, seq_len, seed + step)
        bytes_est = estimate_bytes_per_dst_from_uniform(batch_T, 2, E, experts_per_rank, 4, ctx_dim, world_size)
        
        assignments_fair, _, _ = router_fair(x_t, token_ids, pos_ids, bytes_est, training=True)
        assignments_unfair, _, _ = router_unfair(x_t, token_ids, pos_ids, bytes_est, training=True)
        
        for t, (pos, fair_experts, unfair_experts) in enumerate(zip(pos_ids, assignments_fair, assignments_unfair)):
            pos_window = int((pos.item() * num_windows) // (seq_len + 1))
            position_utilization_fair[pos_window].extend(fair_experts)
            position_utilization_unfair[pos_window].extend(unfair_experts)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    windows = list(range(num_windows))
    fair_counts = [len(position_utilization_fair[w]) for w in windows]
    unfair_counts = [len(position_utilization_unfair[w]) for w in windows]
    
    ax1.bar([w - 0.2 for w in windows], fair_counts, width=0.4, label='With fairness', color='green', alpha=0.7)
    ax1.bar([w + 0.2 for w in windows], unfair_counts, width=0.4, label='Without fairness', color='red', alpha=0.7)
    ax1.set_xlabel('Position window')
    ax1.set_ylabel('Total expert assignments')
    ax1.set_title('Expert Assignments by Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    fair_expert_counts = Counter()
    unfair_expert_counts = Counter()
    for w in windows:
        for expert in position_utilization_fair[w]:
            fair_expert_counts[expert] += 1
        for expert in position_utilization_unfair[w]:
            unfair_expert_counts[expert] += 1
    
    experts = list(range(E))
    fair_util = [fair_expert_counts[e] for e in experts]
    unfair_util = [unfair_expert_counts[e] for e in experts]
    
    ax2.bar([e - 0.2 for e in experts], fair_util, width=0.4, label='With fairness', color='green', alpha=0.7)
    ax2.bar([e + 0.2 for e in experts], unfair_util, width=0.4, label='Without fairness', color='red', alpha=0.7)
    ax2.set_xlabel('Expert ID')
    ax2.set_ylabel('Total utilization')
    ax2.set_title('Expert Utilization Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    late_window_threshold = num_windows // 2
    early_assignments_fair = sum(fair_counts[:late_window_threshold])
    late_assignments_fair = sum(fair_counts[late_window_threshold:])
    early_assignments_unfair = sum(unfair_counts[:late_window_threshold])
    late_assignments_unfair = sum(unfair_counts[late_window_threshold:])
    
    categories = ['Early positions', 'Late positions']
    fair_ratios = [early_assignments_fair, late_assignments_fair]
    unfair_ratios = [early_assignments_unfair, late_assignments_unfair]
    
    x_pos = np.arange(len(categories))
    ax3.bar(x_pos - 0.2, fair_ratios, width=0.4, label='With fairness', color='green', alpha=0.7)
    ax3.bar(x_pos + 0.2, unfair_ratios, width=0.4, label='Without fairness', color='red', alpha=0.7)
    ax3.set_xlabel('Position category')
    ax3.set_ylabel('Total assignments')
    ax3.set_title('Early vs Late Position Assignments')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    fair_variance = np.var(fair_util)
    unfair_variance = np.var(unfair_util)
    
    metrics = ['Utilization\nVariance', 'Late Position\nStarvation']
    fair_metrics = [fair_variance, 1.0 - (late_assignments_fair / max(1, late_assignments_fair + early_assignments_fair))]
    unfair_metrics = [unfair_variance, 1.0 - (late_assignments_unfair / max(1, late_assignments_unfair + early_assignments_unfair))]
    
    x_pos = np.arange(len(metrics))
    ax4.bar(x_pos - 0.2, fair_metrics, width=0.4, label='With fairness', color='green', alpha=0.7)
    ax4.bar(x_pos + 0.2, unfair_metrics, width=0.4, label='Without fairness', color='red', alpha=0.7)
    ax4.set_xlabel('Metric')
    ax4.set_ylabel('Value')
    ax4.set_title('Fairness Metrics Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_position_fairness.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    starvation_reduction = (unfair_metrics[1] - fair_metrics[1]) / max(unfair_metrics[1], 1e-6)
    fairness_score = 1.0 / (1.0 + fair_variance)
    
    print(f"   ✅ Starvation reduction: {starvation_reduction:.1%}")
    print(f"   📊 Fairness score: {fairness_score:.4f}")
    print(f"   📈 Utilization variance: {fair_variance:.4f}")
    
    return {
        "starvation_reduction": starvation_reduction,
        "fairness_score": fairness_score,
        "utilization_variance": fair_variance
    }


def run_exp3_compute_skipping(seed: int = 0,
                             world_size: int = 4,
                             experts_per_rank: int = 2,
                             batch_T: int = 128,
                             ctx_dim: int = 64,
                             vocab_size: int = 1000,
                             seq_len: int = 512,
                             flops_budget: float = 1e6,
                             d_model: int = 64,
                             ffn_dim: int = 256,
                             num_steps: int = 50,
                             output_dir: str = ".") -> Dict[str, float]:
    """
    Experiment 3: Compute-Aware Expert Skipping
    
    Tests the compute-aware skipping mechanism for FLOPs budget control.
    """
    print(f"🔧 Exp3: Compute-aware skipping (FLOPs budget: {flops_budget:.0e})")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    E = world_size * experts_per_rank
    device_map = make_device_map(world_size, experts_per_rank)
    rank2node = [i // 2 for i in range(world_size)]
    tiers = build_tiers_for_rank(0, world_size, experts_per_rank, rank2node)
    
    cost_cache = CostCache(world_size, ema=0.1, device="cpu")
    
    router = BudgetedHierarchicalTopKRouter(
        num_experts=E,
        k=2,
        tiers=tiers,
        device_map=device_map,
        cost_cache=cost_cache,
        flops_budget=flops_budget,
        tau_gains=(0.05, 0.001),
        token_vocab_size=vocab_size,
        ctx_dim=ctx_dim,
        device_id=0,
        device="cpu"
    )
    
    tau_history = []
    flops_history = []
    skip_rates = []
    
    for step in range(num_steps):
        x_t, token_ids, pos_ids = generate_synthetic_batch(batch_T, ctx_dim, vocab_size, seq_len, seed + step)
        bytes_est = estimate_bytes_per_dst_from_uniform(batch_T, 2, E, experts_per_rank, 4, ctx_dim, world_size)
        
        assignments, logits, losses = router(x_t, token_ids, pos_ids, bytes_est, 
                                           training=False, compute_skip=True)
        
        total_flops = estimate_flops_per_token(assignments, d_model, ffn_dim) * batch_T
        flops_history.append(total_flops)
        
        skipped = sum(1 for a in assignments if len(a) == 0)
        skip_rate = skipped / batch_T
        skip_rates.append(skip_rate)
        
        tau = router.update_tau(total_flops)
        tau_history.append(tau)
        
        if step % 10 == 0:
            print(f"   Step {step:2d}: τ={tau:.3f}, FLOPs={total_flops:.0e}, skip_rate={skip_rate:.1%}")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    steps = range(num_steps)
    ax1.plot(steps, tau_history, 'b-', linewidth=2, label='τ (skip threshold)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('τ value')
    ax1.set_title('Compute Skip Controller')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(steps, flops_history, 'r-', linewidth=2, label='Observed FLOPs')
    ax2.axhline(y=flops_budget, color='g', linestyle='--', linewidth=2, label='Budget')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('FLOPs per batch')
    ax2.set_title('FLOPs Budget Tracking')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    ax3.plot(steps, [100 * sr for sr in skip_rates], 'purple', linewidth=2, label='Skip rate')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Skip rate (%)')
    ax3.set_title('Expert Skipping Rate')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    baseline_flops = estimate_flops_per_token([list(range(2)) for _ in range(batch_T)], d_model, ffn_dim) * batch_T
    flops_reduction = [(baseline_flops - f) / baseline_flops for f in flops_history]
    ax4.plot(steps, [100 * fr for fr in flops_reduction], 'orange', linewidth=2, label='FLOPs reduction')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('FLOPs reduction (%)')
    ax4.set_title('Compute Savings')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp3_compute_skipping.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    final_tau = tau_history[-1]
    avg_skip_rate = np.mean(skip_rates[-10:])
    avg_flops_reduction = np.mean(flops_reduction[-10:])
    
    print(f"   ✅ Final τ: {final_tau:.4f}")
    print(f"   📊 Avg skip rate: {avg_skip_rate:.1%}")
    print(f"   🎯 FLOPs reduction: {avg_flops_reduction:.1%}")
    
    return {
        "final_tau": final_tau,
        "avg_skip_rate": avg_skip_rate,
        "flops_reduction": avg_flops_reduction
    }
