#!/usr/bin/env python3
"""
Data preprocessing for BESS++ Energy-Bounded Attention Experiment
Generates synthetic datasets with different attention patterns
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
import math

def generate_synthetic_data(
    seq_lengths: List[int] = [512, 1024, 2048],
    d_model: int = 512,
    d_v: int = 512,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generate synthetic Q, K, V tensors with different attention patterns
    for testing BESS++ algorithm effectiveness.
    
    Returns:
        Dictionary mapping pattern names to (Q, K, V) tuples
    """
    
    patterns = {}
    
    T = max(seq_lengths)
    
    print(f"Generating synthetic data: T={T}, d_model={d_model}, d_v={d_v}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    Q_gauss = torch.randn(T, d_model, device=device, dtype=dtype)
    K_gauss = torch.randn(T, d_model, device=device, dtype=dtype)
    V_gauss = torch.randn(T, d_v, device=device, dtype=dtype)
    patterns['gaussian'] = (Q_gauss, K_gauss, V_gauss)
    
    rank = min(64, d_model // 8)
    A = torch.randn(T, rank, device=device, dtype=dtype)
    B_k = torch.randn(rank, d_model, device=device, dtype=dtype)
    B_q = torch.randn(rank, d_model, device=device, dtype=dtype)
    B_v = torch.randn(rank, d_v, device=device, dtype=dtype)
    
    Q_lr = A @ B_q + 0.1 * torch.randn(T, d_model, device=device, dtype=dtype)
    K_lr = A @ B_k + 0.1 * torch.randn(T, d_model, device=device, dtype=dtype)
    V_lr = A @ B_v + 0.1 * torch.randn(T, d_v, device=device, dtype=dtype)
    patterns['lowrank'] = (Q_lr, K_lr, V_lr)
    
    Q_spiky = torch.randn(T, d_model, device=device, dtype=dtype)
    K_spiky = torch.randn(T, d_model, device=device, dtype=dtype)
    V_spiky = torch.randn(T, d_v, device=device, dtype=dtype)
    
    n_spikes = max(8, T // 64)
    spike_indices = torch.randperm(T, device=device)[:n_spikes]
    K_spiky[spike_indices] *= 3.0  # Amplify spike keys
    V_spiky[spike_indices] *= 2.0  # Amplify corresponding values
    patterns['spiky'] = (Q_spiky, K_spiky, V_spiky)
    
    Q_causal = torch.randn(T, d_model, device=device, dtype=dtype)
    K_causal = torch.randn(T, d_model, device=device, dtype=dtype)
    V_causal = torch.randn(T, d_v, device=device, dtype=dtype)
    
    pos_embed = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)
    pos_sim = torch.exp(-0.01 * torch.abs(pos_embed - pos_embed.T))
    
    for i in range(T):
        local_weight = pos_sim[i].unsqueeze(1)
        K_causal[i] = K_causal[i] + 0.5 * (local_weight * K_causal).sum(dim=0) / local_weight.sum()
    
    patterns['causal'] = (Q_causal, K_causal, V_causal)
    
    Q_block = torch.randn(T, d_model, device=device, dtype=dtype)
    K_block = torch.randn(T, d_model, device=device, dtype=dtype)
    V_block = torch.randn(T, d_v, device=device, dtype=dtype)
    
    block_size = 64
    n_blocks = T // block_size
    
    for b in range(n_blocks):
        start_idx = b * block_size
        end_idx = min((b + 1) * block_size, T)
        
        block_mean_k = K_block[start_idx:end_idx].mean(dim=0, keepdim=True)
        block_mean_v = V_block[start_idx:end_idx].mean(dim=0, keepdim=True)
        
        K_block[start_idx:end_idx] += 0.3 * block_mean_k
        V_block[start_idx:end_idx] += 0.3 * block_mean_v
    
    patterns['block_sparse'] = (Q_block, K_block, V_block)
    
    print(f"Generated {len(patterns)} attention patterns: {list(patterns.keys())}")
    
    return patterns

def prepare_datasets(
    seq_lengths: List[int] = [512, 1024, 2048],
    batch_sizes: List[int] = [1, 2, 4],
    device: str = 'cuda'
) -> Tuple[List, List]:
    """
    Prepare training and validation datasets for the BESS++ model.
    
    Returns:
        Tuple of (train_data, val_data) lists containing data configurations
    """
    
    train_data = []
    val_data = []
    
    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            train_config = {
                'seq_length': seq_len,
                'batch_size': batch_size,
                'd_model': 512,
                'd_v': 512,
                'device': device
            }
            train_data.append(train_config)
            
            val_config = train_config.copy()
            val_config['batch_size'] = min(batch_size, 2)
            val_data.append(val_config)
    
    print(f"Prepared {len(train_data)} training and {len(val_data)} validation configurations")
    
    return train_data, val_data

def create_attention_mask(seq_length: int, mask_type: str = 'causal', device: str = 'cuda') -> torch.Tensor:
    """
    Create attention masks for different attention patterns.
    
    Args:
        seq_length: Sequence length
        mask_type: Type of mask ('causal', 'full', 'local', 'block')
        device: Device to create tensor on
        
    Returns:
        Boolean mask tensor of shape [seq_length, seq_length]
    """
    
    if mask_type == 'causal':
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device, dtype=torch.bool))
    
    elif mask_type == 'full':
        mask = torch.ones(seq_length, seq_length, device=device, dtype=torch.bool)
    
    elif mask_type == 'local':
        window_size = min(128, seq_length // 4)
        mask = torch.zeros(seq_length, seq_length, device=device, dtype=torch.bool)
        
        for i in range(seq_length):
            start = max(0, i - window_size // 2)
            end = min(seq_length, i + window_size // 2 + 1)
            mask[i, start:end] = True
    
    elif mask_type == 'block':
        block_size = 64
        mask = torch.zeros(seq_length, seq_length, device=device, dtype=torch.bool)
        
        for i in range(0, seq_length, block_size):
            end_i = min(i + block_size, seq_length)
            for j in range(0, seq_length, block_size):
                end_j = min(j + block_size, seq_length)
                if j <= i:
                    mask[i:end_i, j:end_j] = True
    
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    
    return mask

def compute_attention_statistics(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about attention patterns for analysis.
    
    Args:
        Q, K, V: Query, Key, Value tensors
        
    Returns:
        Dictionary of attention statistics
    """
    
    with torch.no_grad():
        logits = Q @ K.T  # [T, T]
        
        attn_probs = torch.softmax(logits, dim=-1)
        
        stats = {
            'max_attention': float(attn_probs.max()),
            'min_attention': float(attn_probs.min()),
            'attention_entropy': float(-(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean()),
            'attention_sparsity': float((attn_probs < 0.01).float().mean()),
            'key_norm_max': float(K.norm(dim=-1).max()),
            'key_norm_min': float(K.norm(dim=-1).min()),
            'query_norm_max': float(Q.norm(dim=-1).max()),
            'query_norm_min': float(Q.norm(dim=-1).min()),
            'value_norm_max': float(V.norm(dim=-1).max()),
            'value_norm_min': float(V.norm(dim=-1).min()),
        }
    
    return stats

def normalize_qkv(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                  scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize Q, K, V tensors for stable attention computation.
    
    Args:
        Q, K, V: Input tensors
        scale: Scaling factor (default: 1/sqrt(d_model))
        
    Returns:
        Normalized Q, K, V tensors
    """
    
    d_model = Q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d_model)
    
    Q_norm = Q * scale
    
    K_norm = K / K.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    V_norm = V / V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    return Q_norm, K_norm, V_norm

if __name__ == "__main__":
    print("Testing synthetic data generation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patterns = generate_synthetic_data(device=device)
    
    for name, (Q, K, V) in patterns.items():
        stats = compute_attention_statistics(Q, K, V)
        print(f"\nPattern: {name}")
        print(f"  Shape: Q={Q.shape}, K={K.shape}, V={V.shape}")
        print(f"  Attention entropy: {stats['attention_entropy']:.3f}")
        print(f"  Attention sparsity: {stats['attention_sparsity']:.3f}")
        print(f"  Key norm range: [{stats['key_norm_min']:.3f}, {stats['key_norm_max']:.3f}]")
    
    print("\n✓ Data preprocessing module test completed")
