"""
Data preprocessing module for RQ-Stream Router experiments.
This module contains data preparation and synthetic data generation utilities.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Iterator
import random


def generate_synthetic_tokens(vocab_size: int, 
                             seq_len: int, 
                             batch_size: int = 1,
                             n_batches: int = 100,
                             device: str = 'cpu') -> Iterator[torch.Tensor]:
    """
    Generate synthetic token sequences for language model training/evaluation.
    
    Args:
        vocab_size: Size of vocabulary
        seq_len: Length of each sequence
        batch_size: Number of sequences per batch
        n_batches: Number of batches to generate
        device: Device to place tensors on
    
    Yields:
        Batches of token sequences
    """
    for _ in range(n_batches):
        batch = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        yield batch


def generate_markovian_codes(n_tokens: int,
                            n_codes: int,
                            self_transition_prob: float = 0.95,
                            seed: Optional[int] = None) -> List[int]:
    """
    Generate Markovian code sequences for routing experiments.
    
    Args:
        n_tokens: Number of tokens to generate
        n_codes: Number of possible codes
        self_transition_prob: Probability of staying in same code
        seed: Random seed for reproducibility
    
    Returns:
        List of code assignments
    """
    if seed is not None:
        random.seed(seed)
    
    codes = []
    current_code = random.randrange(n_codes)
    
    for _ in range(n_tokens):
        codes.append(current_code)
        
        if random.random() > self_transition_prob:
            current_code = random.randrange(n_codes)
    
    return codes


def generate_bursty_demand_pattern(n_windows: int,
                                  n_codes: int,
                                  burst_probability: float = 0.3,
                                  burst_intensity: float = 5.0,
                                  background_level: float = 1.0,
                                  seed: Optional[int] = None) -> List[torch.Tensor]:
    """
    Generate bursty demand patterns for capacity scheduling experiments.
    
    Args:
        n_windows: Number of time windows
        n_codes: Number of routing codes
        burst_probability: Probability of burst in each window
        burst_intensity: Intensity multiplier for burst codes
        background_level: Background demand level
        seed: Random seed for reproducibility
    
    Returns:
        List of demand tensors for each window
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    demand_patterns = []
    
    for w in range(n_windows):
        demand = torch.rand(n_codes) * background_level + background_level
        
        if random.random() < burst_probability:
            n_burst_codes = random.randint(1, min(3, n_codes))
            burst_codes = random.sample(range(n_codes), n_burst_codes)
            
            for code_idx in burst_codes:
                burst_magnitude = random.uniform(burst_intensity, burst_intensity * 2)
                demand[code_idx] += burst_magnitude
        
        demand_patterns.append(demand)
    
    return demand_patterns


def create_streaming_assignments(codes: List[int],
                                expert_indices: torch.Tensor,
                                window_size: int) -> List[List[Tuple[int, int, Optional[int]]]]:
    """
    Create streaming assignments for communication simulation.
    
    Args:
        codes: List of routing codes
        expert_indices: Tensor of expert assignments [T, k]
        window_size: Size of processing windows
    
    Returns:
        List of assignment lists for each window
    """
    assignments_by_window = []
    
    for w_start in range(0, len(codes), window_size):
        w_end = min(w_start + window_size, len(codes))
        window_assignments = []
        
        for t in range(w_start, w_end):
            code = codes[t]
            t_local = t - w_start
            
            if t_local < expert_indices.size(0):
                experts = expert_indices[t_local].tolist()
                primary = experts[0] if len(experts) > 0 else 0
                secondary = experts[1] if len(experts) > 1 else None
            else:
                primary = 0
                secondary = None
            
            window_assignments.append((code, primary, secondary))
        
        assignments_by_window.append(window_assignments)
    
    return assignments_by_window


def prepare_ssm_states(batch_size: int,
                      seq_len: int,
                      state_dim: int,
                      device: str = 'cpu') -> torch.Tensor:
    """
    Prepare synthetic SSM states for RQ-Stream routing.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        state_dim: State dimension
        device: Device to place tensor on
    
    Returns:
        Synthetic SSM state tensor
    """
    states = torch.randn(batch_size, seq_len, state_dim, device=device)
    
    alpha = 0.9  # Smoothing factor
    for t in range(1, seq_len):
        states[:, t] = alpha * states[:, t-1] + (1 - alpha) * states[:, t]
    
    return states


def create_code_history_windows(code_patterns: List[torch.Tensor],
                               window_size: int = 5) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Create sliding windows of code histories for Router-SSM training.
    
    Args:
        code_patterns: List of code demand patterns
        window_size: Size of history window
    
    Returns:
        Tuple of (input_windows, target_patterns)
    """
    input_windows = []
    targets = []
    
    for i in range(window_size, len(code_patterns)):
        window = torch.stack(code_patterns[i-window_size:i])  # [window_size, n_codes]
        input_windows.append(window)
        
        targets.append(code_patterns[i])
    
    return input_windows, targets


def normalize_demand_patterns(patterns: List[torch.Tensor],
                             method: str = 'minmax') -> List[torch.Tensor]:
    """
    Normalize demand patterns for better training stability.
    
    Args:
        patterns: List of demand pattern tensors
        method: Normalization method ('minmax', 'zscore', 'none')
    
    Returns:
        List of normalized patterns
    """
    if method == 'none':
        return patterns
    
    all_values = torch.cat(patterns, dim=0)
    
    if method == 'minmax':
        min_val = all_values.min()
        max_val = all_values.max()
        range_val = max_val - min_val
        
        if range_val > 1e-8:
            normalized = [(p - min_val) / range_val for p in patterns]
        else:
            normalized = patterns
    
    elif method == 'zscore':
        mean_val = all_values.mean()
        std_val = all_values.std()
        
        if std_val > 1e-8:
            normalized = [(p - mean_val) / std_val for p in patterns]
        else:
            normalized = patterns
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def split_data(data: List,
               train_ratio: float = 0.8,
               val_ratio: float = 0.1,
               test_ratio: float = 0.1,
               seed: Optional[int] = None) -> Tuple[List, List, List]:
    """
    Split data into train/validation/test sets.
    
    Args:
        data: List of data samples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    if seed is not None:
        random.seed(seed)
    
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    n_total = len(shuffled_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = shuffled_data[:n_train]
    val_data = shuffled_data[n_train:n_train + n_val]
    test_data = shuffled_data[n_train + n_val:]
    
    return train_data, val_data, test_data


def create_capacity_constraints(n_experts: int,
                               base_capacity: int,
                               capacity_variation: float = 0.2,
                               seed: Optional[int] = None) -> Dict[int, int]:
    """
    Create heterogeneous capacity constraints for experts.
    
    Args:
        n_experts: Number of experts
        base_capacity: Base capacity per expert
        capacity_variation: Relative variation in capacities
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping expert ID to capacity
    """
    if seed is not None:
        random.seed(seed)
    
    capacities = {}
    
    for e in range(n_experts):
        variation = random.uniform(-capacity_variation, capacity_variation)
        capacity = int(base_capacity * (1 + variation))
        capacity = max(1, capacity)  # Ensure positive capacity
        capacities[e] = capacity
    
    return capacities


def preprocess_for_experiment(experiment_type: str,
                             **kwargs) -> Dict:
    """
    Preprocess data for specific experiment types.
    
    Args:
        experiment_type: Type of experiment ('streaming', 'lm_quality', 'capacity')
        **kwargs: Experiment-specific parameters
    
    Returns:
        Dictionary of preprocessed data
    """
    if experiment_type == 'streaming':
        window_sizes = kwargs.get('window_sizes', [64, 128, 256])
        n_codes = kwargs.get('n_codes', 64)
        n_experts = kwargs.get('n_experts', 16)
        
        data = {
            'window_sizes': window_sizes,
            'synthetic_codes': {},
            'capacity_constraints': {}
        }
        
        for W in window_sizes:
            codes = generate_markovian_codes(W, n_codes, seed=42)
            data['synthetic_codes'][W] = codes
            data['capacity_constraints'][W] = create_capacity_constraints(n_experts, W//4, seed=42)
        
        return data
    
    elif experiment_type == 'lm_quality':
        vocab_size = kwargs.get('vocab_size', 1000)
        seq_len = kwargs.get('seq_len', 128)
        batch_size = kwargs.get('batch_size', 4)
        n_batches = kwargs.get('n_batches', 100)
        
        data = {
            'train_batches': list(generate_synthetic_tokens(vocab_size, seq_len, batch_size, n_batches)),
            'val_batches': list(generate_synthetic_tokens(vocab_size, seq_len, batch_size, n_batches//5)),
            'vocab_size': vocab_size,
            'seq_len': seq_len
        }
        
        return data
    
    elif experiment_type == 'capacity':
        n_windows = kwargs.get('n_windows', 50)
        n_codes = kwargs.get('n_codes', 64)
        
        demand_patterns = generate_bursty_demand_pattern(n_windows, n_codes, seed=42)
        input_windows, targets = create_code_history_windows(demand_patterns)
        
        data = {
            'demand_patterns': demand_patterns,
            'input_windows': input_windows,
            'targets': targets,
            'n_codes': n_codes
        }
        
        return data
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
