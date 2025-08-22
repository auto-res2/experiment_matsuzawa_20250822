"""
Training module for RQ-Stream Router experiments.
This module contains training utilities and functions used by the main experimental pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


def train_tiny_lm(model: nn.Module, 
                  optimizer: torch.optim.Optimizer,
                  data_loader: Optional[object] = None,
                  n_steps: int = 100,
                  vocab_size: int = 1000,
                  seq_len: int = 128,
                  batch_size: int = 4,
                  device: str = 'cpu') -> List[float]:
    """
    Train a tiny language model for experimental purposes.
    
    Args:
        model: The language model to train
        optimizer: Optimizer for training
        data_loader: Optional data loader (if None, uses synthetic data)
        n_steps: Number of training steps
        vocab_size: Vocabulary size for synthetic data
        seq_len: Sequence length for synthetic data
        batch_size: Batch size for synthetic data
        device: Device to train on
    
    Returns:
        List of training losses
    """
    model.train()
    losses = []
    
    def generate_synthetic_batch():
        return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"Training model for {n_steps} steps...")
    
    for step in range(n_steps):
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
        
        optimizer.zero_grad()
        logits = model(input_ids)
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            targets.reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 20 == 0:
            print(f"Step {step+1}/{n_steps}: Loss = {loss.item():.4f}")
    
    return losses


def train_router_ssm(model: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     code_histories: List[torch.Tensor],
                     n_epochs: int = 20,
                     window_size: int = 5) -> List[float]:
    """
    Train the Router-SSM predictor for capacity scheduling.
    
    Args:
        model: Router-SSM model to train
        optimizer: Optimizer for training
        code_histories: List of code histograms for training
        n_epochs: Number of training epochs
        window_size: Size of history window for prediction
    
    Returns:
        List of training losses
    """
    model.train()
    losses = []
    
    print(f"Training Router-SSM for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_samples = 0
        
        for i in range(window_size, len(code_histories) - 1):
            hist_window = torch.stack(code_histories[i-window_size:i]).unsqueeze(0)  # [1, window_size, K]
            target = code_histories[i+1]  # [K]
            
            optimizer.zero_grad()
            pred = model(hist_window).squeeze(0)  # [K]
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_samples += 1
        
        avg_loss = total_loss / max(1, n_samples)
        losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: MSE Loss = {avg_loss:.4f}")
    
    return losses


def compute_training_metrics(losses: List[float]) -> Dict[str, float]:
    """
    Compute training metrics from loss history.
    
    Args:
        losses: List of training losses
    
    Returns:
        Dictionary of computed metrics
    """
    if not losses:
        return {}
    
    losses_array = np.array(losses)
    
    metrics = {
        'final_loss': float(losses[-1]),
        'min_loss': float(np.min(losses_array)),
        'mean_loss': float(np.mean(losses_array)),
        'loss_std': float(np.std(losses_array)),
        'convergence_rate': float((losses[0] - losses[-1]) / max(losses[0], 1e-8))
    }
    
    return metrics


def setup_optimizer(model: nn.Module, 
                   lr: float = 1e-3, 
                   weight_decay: float = 0.01,
                   optimizer_type: str = 'adam') -> torch.optim.Optimizer:
    """
    Setup optimizer for model training.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
    
    Returns:
        Configured optimizer
    """
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_lr_scheduler(optimizer: torch.optim.Optimizer,
                       scheduler_type: str = 'cosine',
                       n_steps: int = 1000) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('cosine', 'step', 'none')
        n_steps: Total number of training steps
    
    Returns:
        Learning rate scheduler or None
    """
    if scheduler_type.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    elif scheduler_type.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_steps//3, gamma=0.5)
    elif scheduler_type.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
