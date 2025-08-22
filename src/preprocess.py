"""
FOST-PEFT Data Preprocessing Module
Handles synthetic data generation and real dataset preparation for continual learning experiments.
"""

import os
import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed: int = 0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_synthetic_stream(n_tasks: int = 5, samples_per_task: int = 1000, 
                            input_dim: int = 128, n_classes: int = 10, 
                            drift_strength: float = 0.3, seed: int = 0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate synthetic continual learning stream with concept drift.
    
    Args:
        n_tasks: Number of tasks in the stream
        samples_per_task: Samples per task
        input_dim: Input feature dimension
        n_classes: Number of classes
        drift_strength: Strength of concept drift between tasks
        seed: Random seed
        
    Returns:
        List of (features, labels) tuples for each task
    """
    set_seed(seed)
    
    tasks = []
    base_centers = torch.randn(n_classes, input_dim) * 2.0
    
    for task_id in range(n_tasks):
        drift_noise = torch.randn_like(base_centers) * drift_strength * task_id
        task_centers = base_centers + drift_noise
        
        features = []
        labels = []
        
        for class_id in range(n_classes):
            n_samples = samples_per_task // n_classes
            if class_id < samples_per_task % n_classes:
                n_samples += 1
                
            class_features = task_centers[class_id].unsqueeze(0) + torch.randn(n_samples, input_dim) * 0.5
            class_labels = torch.full((n_samples,), class_id, dtype=torch.long)
            
            features.append(class_features)
            labels.append(class_labels)
        
        task_features = torch.cat(features, dim=0)
        task_labels = torch.cat(labels, dim=0)
        
        perm = torch.randperm(len(task_features))
        task_features = task_features[perm]
        task_labels = task_labels[perm]
        
        tasks.append((task_features, task_labels))
    
    return tasks


def create_dataloaders(tasks: List[Tuple[torch.Tensor, torch.Tensor]], 
                      batch_size: int = 32, shuffle: bool = True) -> List[DataLoader]:
    """Create DataLoaders from task data."""
    dataloaders = []
    
    for features, labels in tasks:
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataloaders.append(dataloader)
    
    return dataloaders


def prepare_cifar_stream(data_dir: str = "./data", n_tasks: int = 5) -> List[DataLoader]:
    """
    Prepare CIFAR-100 class-incremental stream (placeholder for real implementation).
    For the quick test, this returns synthetic data.
    """
    print("Note: Using synthetic data for quick test. Real CIFAR-100 implementation would go here.")
    tasks = generate_synthetic_stream(n_tasks=n_tasks, samples_per_task=200, 
                                    input_dim=512, n_classes=20)
    return create_dataloaders(tasks, batch_size=16)


def prepare_text_stream(data_dir: str = "./data", n_tasks: int = 3) -> List[DataLoader]:
    """
    Prepare text domain adaptation stream (placeholder for real implementation).
    For the quick test, this returns synthetic data.
    """
    print("Note: Using synthetic data for quick test. Real text implementation would go here.")
    tasks = generate_synthetic_stream(n_tasks=n_tasks, samples_per_task=150, 
                                    input_dim=768, n_classes=5)
    return create_dataloaders(tasks, batch_size=8)


if __name__ == "__main__":
    print("Testing synthetic data generation...")
    tasks = generate_synthetic_stream(n_tasks=3, samples_per_task=100, input_dim=64, n_classes=5)
    dataloaders = create_dataloaders(tasks, batch_size=16)
    
    print(f"Generated {len(tasks)} tasks")
    for i, (features, labels) in enumerate(tasks):
        print(f"Task {i}: {features.shape} features, {labels.shape} labels")
        print(f"  Classes: {torch.unique(labels).tolist()}")
    
    print("Preprocessing test completed successfully!")
