"""
Evaluation module for RQ-Stream Router experiments.
This module contains evaluation utilities and metrics computation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import math


def evaluate_language_model(model: nn.Module,
                           data_loader: Optional[object] = None,
                           n_batches: int = 20,
                           vocab_size: int = 1000,
                           seq_len: int = 128,
                           batch_size: int = 4,
                           device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate a language model and compute perplexity and other metrics.
    
    Args:
        model: Language model to evaluate
        data_loader: Optional data loader (if None, uses synthetic data)
        n_batches: Number of evaluation batches
        vocab_size: Vocabulary size for synthetic data
        seq_len: Sequence length for synthetic data
        batch_size: Batch size for synthetic data
        device: Device to evaluate on
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    losses = []
    
    def generate_synthetic_batch():
        return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"Evaluating model on {n_batches} batches...")
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            if data_loader is None:
                input_ids = generate_synthetic_batch()
            else:
                try:
                    input_ids = next(iter(data_loader))
                    if isinstance(input_ids, (list, tuple)):
                        input_ids = input_ids[0]
                    input_ids = input_ids.to(device)
                except:
                    input_ids = generate_synthetic_batch()
            
            targets = torch.roll(input_ids, -1, dims=1)
            
            logits = model(input_ids)
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1),
                reduction='sum'
            )
            
            batch_tokens = input_ids.numel()
            total_loss += loss.item()
            total_tokens += batch_tokens
            losses.append(loss.item() / batch_tokens)
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    metrics = {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
        'loss_std': float(np.std(losses)) if losses else 0.0
    }
    
    print(f"Evaluation results: Loss = {avg_loss:.4f}, Perplexity = {perplexity:.2f}")
    
    return metrics


def evaluate_routing_efficiency(router_stats: List[Dict],
                               baseline_stats: List[Dict]) -> Dict[str, float]:
    """
    Evaluate routing efficiency by comparing RQ-Stream vs baseline routing.
    
    Args:
        router_stats: List of routing statistics from RQ-Stream router
        baseline_stats: List of routing statistics from baseline router
    
    Returns:
        Dictionary of efficiency metrics
    """
    if not router_stats or not baseline_stats:
        return {}
    
    rq_messages = [s.get('num_messages', 0) for s in router_stats]
    baseline_messages = [s.get('num_messages', 0) for s in baseline_stats]
    
    rq_msg_sizes = [s.get('avg_msg_size', 0) for s in router_stats]
    baseline_msg_sizes = [s.get('avg_msg_size', 0) for s in baseline_stats]
    
    rq_latency = [s.get('latency_us', 0) for s in router_stats]
    baseline_latency = [s.get('latency_us', 0) for s in baseline_stats]
    
    rq_switches = [s.get('route_switches_per_100', 0) for s in router_stats]
    baseline_switches = [s.get('route_switches_per_100', 0) for s in baseline_stats]
    
    avg_rq_messages = np.mean(rq_messages)
    avg_baseline_messages = np.mean(baseline_messages)
    message_reduction = (avg_baseline_messages - avg_rq_messages) / max(avg_baseline_messages, 1e-8)
    
    avg_rq_size = np.mean(rq_msg_sizes)
    avg_baseline_size = np.mean(baseline_msg_sizes)
    size_increase = (avg_rq_size - avg_baseline_size) / max(avg_baseline_size, 1e-8)
    
    avg_rq_latency = np.mean(rq_latency)
    avg_baseline_latency = np.mean(baseline_latency)
    latency_reduction = (avg_baseline_latency - avg_rq_latency) / max(avg_baseline_latency, 1e-8)
    
    avg_rq_switches = np.mean(rq_switches)
    avg_baseline_switches = np.mean(baseline_switches)
    switch_reduction = (avg_baseline_switches - avg_rq_switches) / max(avg_baseline_switches, 1e-8)
    
    metrics = {
        'message_fragmentation_reduction': message_reduction,
        'message_size_increase': size_increase,
        'latency_reduction': latency_reduction,
        'route_switch_reduction': switch_reduction,
        'avg_rq_messages': avg_rq_messages,
        'avg_baseline_messages': avg_baseline_messages,
        'avg_rq_msg_size': avg_rq_size,
        'avg_baseline_msg_size': avg_baseline_size
    }
    
    return metrics


def evaluate_capacity_scheduling(predictions: List[torch.Tensor],
                                targets: List[torch.Tensor],
                                uniform_overflows: List[float],
                                adaptive_overflows: List[float]) -> Dict[str, float]:
    """
    Evaluate capacity scheduling effectiveness.
    
    Args:
        predictions: List of predicted demand tensors
        targets: List of target demand tensors
        uniform_overflows: List of overflow values with uniform allocation
        adaptive_overflows: List of overflow values with adaptive allocation
    
    Returns:
        Dictionary of scheduling metrics
    """
    if not predictions or not targets:
        return {}
    
    prediction_errors = []
    for pred, target in zip(predictions, targets):
        mae = F.l1_loss(pred, target).item()
        prediction_errors.append(mae)
    
    avg_mae = np.mean(prediction_errors)
    
    avg_uniform_overflow = np.mean(uniform_overflows) if uniform_overflows else 0.0
    avg_adaptive_overflow = np.mean(adaptive_overflows) if adaptive_overflows else 0.0
    
    overflow_reduction = (avg_uniform_overflow - avg_adaptive_overflow) / max(avg_uniform_overflow, 1e-8)
    
    metrics = {
        'prediction_mae': avg_mae,
        'uniform_overflow': avg_uniform_overflow,
        'adaptive_overflow': avg_adaptive_overflow,
        'overflow_reduction': overflow_reduction,
        'prediction_std': float(np.std(prediction_errors)) if prediction_errors else 0.0
    }
    
    return metrics


def compute_gating_flops(E: int, r: int, sequence_length: int) -> Dict[str, int]:
    """
    Compute gating FLOPs for baseline vs RQ-Stream routing.
    
    Args:
        E: Number of experts
        r: Shortlist size for RQ-Stream
        sequence_length: Length of input sequence
    
    Returns:
        Dictionary of FLOP counts
    """
    baseline_flops = E * sequence_length
    
    rq_flops = r * sequence_length
    
    flop_reduction = (baseline_flops - rq_flops) / baseline_flops
    
    return {
        'baseline_flops': baseline_flops,
        'rq_stream_flops': rq_flops,
        'flop_reduction': flop_reduction
    }


def evaluate_model_stability(losses: List[float], 
                            window_size: int = 10) -> Dict[str, float]:
    """
    Evaluate training stability based on loss variance.
    
    Args:
        losses: List of training losses
        window_size: Window size for computing rolling statistics
    
    Returns:
        Dictionary of stability metrics
    """
    if len(losses) < window_size:
        return {'stability_score': 0.0}
    
    losses_array = np.array(losses)
    
    rolling_vars = []
    for i in range(window_size, len(losses)):
        window_losses = losses_array[i-window_size:i]
        rolling_vars.append(np.var(window_losses))
    
    avg_variance = np.mean(rolling_vars)
    stability_score = 1.0 / (1.0 + avg_variance)
    
    if len(losses) >= 2:
        trend = (losses[-1] - losses[0]) / max(abs(losses[0]), 1e-8)
    else:
        trend = 0.0
    
    metrics = {
        'stability_score': stability_score,
        'average_variance': avg_variance,
        'convergence_trend': trend,
        'final_loss': losses[-1] if losses else 0.0
    }
    
    return metrics


def compare_models(baseline_metrics: Dict[str, float],
                  rq_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Compare baseline and RQ-Stream model performance.
    
    Args:
        baseline_metrics: Metrics from baseline model
        rq_metrics: Metrics from RQ-Stream model
    
    Returns:
        Dictionary of comparison metrics
    """
    comparison = {}
    
    if 'perplexity' in baseline_metrics and 'perplexity' in rq_metrics:
        ppl_change = (rq_metrics['perplexity'] - baseline_metrics['perplexity']) / baseline_metrics['perplexity']
        comparison['perplexity_change'] = ppl_change
    
    if 'loss' in baseline_metrics and 'loss' in rq_metrics:
        loss_change = (rq_metrics['loss'] - baseline_metrics['loss']) / baseline_metrics['loss']
        comparison['loss_change'] = loss_change
    
    if 'stability_score' in baseline_metrics and 'stability_score' in rq_metrics:
        stability_improvement = rq_metrics['stability_score'] - baseline_metrics['stability_score']
        comparison['stability_improvement'] = stability_improvement
    
    return comparison
