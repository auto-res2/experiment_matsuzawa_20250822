"""
BEMeGA Preprocessing Module
Generates synthetic few-shot episodes with varying anisotropy, multimodality, and domain shift
"""

import math
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SyntheticEpisodeGenerator:
    """Generate synthetic few-shot episodes with controllable properties"""
    
    def __init__(self, D: int = 128, device: str = "cuda"):
        self.D = D
        self.device = device
        
    def generate_anisotropic_episode(self, N: int, k: int, q: int, 
                                   anisotropy_factor: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate episode with anisotropic class distributions"""
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_idx in range(N):
            U = torch.randn(self.D, self.D, device=self.device)
            U, _ = torch.linalg.qr(U)
            
            eigenvals = torch.ones(self.D, device=self.device)
            n_aniso = min(10, self.D // 4)  # Make first few dimensions highly variant
            eigenvals[:n_aniso] *= anisotropy_factor
            eigenvals[n_aniso:] *= (1.0 / anisotropy_factor)
            
            class_mean = torch.randn(self.D, device=self.device) * 2.0
            
            for _ in range(k):
                noise = torch.randn(self.D, device=self.device)
                noise = noise * torch.sqrt(eigenvals)
                sample = class_mean + U @ noise
                support_data.append(sample)
                support_labels.append(class_idx)
                
            for _ in range(q):
                noise = torch.randn(self.D, device=self.device)
                noise = noise * torch.sqrt(eigenvals)
                sample = class_mean + U @ noise
                query_data.append(sample)
                query_labels.append(class_idx)
        
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels, device=self.device)
        query_data = torch.stack(query_data)
        query_labels = torch.tensor(query_labels, device=self.device)
        
        return support_data, support_labels, query_data, query_labels
    
    def generate_multimodal_episode(self, N: int, k: int, q: int, 
                                  n_modes: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate episode with multimodal class distributions"""
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_idx in range(N):
            modes = []
            for _ in range(n_modes):
                mode_center = torch.randn(self.D, device=self.device) * 3.0
                modes.append(mode_center)
            
            for _ in range(k):
                mode_idx = random.randint(0, n_modes - 1)
                noise = torch.randn(self.D, device=self.device) * 0.5
                sample = modes[mode_idx] + noise
                support_data.append(sample)
                support_labels.append(class_idx)
                
            for _ in range(q):
                mode_idx = random.randint(0, n_modes - 1)
                noise = torch.randn(self.D, device=self.device) * 0.5
                sample = modes[mode_idx] + noise
                query_data.append(sample)
                query_labels.append(class_idx)
        
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels, device=self.device)
        query_data = torch.stack(query_data)
        query_labels = torch.tensor(query_labels, device=self.device)
        
        return support_data, support_labels, query_data, query_labels
    
    def generate_domain_shift_episode(self, N: int, k: int, q: int, 
                                    shift_factor: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate episode with domain shift between support and query"""
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        shift_matrix = torch.randn(self.D, self.D, device=self.device) * 0.1
        shift_matrix = shift_matrix + torch.eye(self.D, device=self.device)
        shift_bias = torch.randn(self.D, device=self.device) * shift_factor
        
        for class_idx in range(N):
            class_mean = torch.randn(self.D, device=self.device) * 2.0
            
            for _ in range(k):
                noise = torch.randn(self.D, device=self.device) * 0.8
                sample = class_mean + noise
                support_data.append(sample)
                support_labels.append(class_idx)
                
            for _ in range(q):
                noise = torch.randn(self.D, device=self.device) * 0.8
                sample = class_mean + noise
                sample = shift_matrix @ sample + shift_bias
                query_data.append(sample)
                query_labels.append(class_idx)
        
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels, device=self.device)
        query_data = torch.stack(query_data)
        query_labels = torch.tensor(query_labels, device=self.device)
        
        return support_data, support_labels, query_data, query_labels
    
    def generate_standard_episode(self, N: int, k: int, q: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate standard isotropic episode for baseline comparison"""
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_idx in range(N):
            class_mean = torch.randn(self.D, device=self.device) * 2.0
            
            for _ in range(k):
                noise = torch.randn(self.D, device=self.device) * 0.8
                sample = class_mean + noise
                support_data.append(sample)
                support_labels.append(class_idx)
                
            for _ in range(q):
                noise = torch.randn(self.D, device=self.device) * 0.8
                sample = class_mean + noise
                query_data.append(sample)
                query_labels.append(class_idx)
        
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels, device=self.device)
        query_data = torch.stack(query_data)
        query_labels = torch.tensor(query_labels, device=self.device)
        
        return support_data, support_labels, query_data, query_labels


def create_episode_batch(generator: SyntheticEpisodeGenerator, 
                        episode_type: str, 
                        N: int, k: int, q: int, 
                        batch_size: int = 100,
                        **kwargs) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Create a batch of episodes of specified type"""
    episodes = []
    
    for _ in range(batch_size):
        if episode_type == "anisotropic":
            episode = generator.generate_anisotropic_episode(N, k, q, **kwargs)
        elif episode_type == "multimodal":
            episode = generator.generate_multimodal_episode(N, k, q, **kwargs)
        elif episode_type == "domain_shift":
            episode = generator.generate_domain_shift_episode(N, k, q, **kwargs)
        elif episode_type == "standard":
            episode = generator.generate_standard_episode(N, k, q)
        else:
            raise ValueError(f"Unknown episode type: {episode_type}")
        
        episodes.append(episode)
    
    return episodes


if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    generator = SyntheticEpisodeGenerator(D=128, device=device)
    
    episode_types = ["standard", "anisotropic", "multimodal", "domain_shift"]
    
    for ep_type in episode_types:
        print(f"\nTesting {ep_type} episodes:")
        if ep_type == "anisotropic":
            episodes = create_episode_batch(generator, ep_type, N=5, k=5, q=15, 
                                          batch_size=10, anisotropy_factor=3.0)
        elif ep_type == "multimodal":
            episodes = create_episode_batch(generator, ep_type, N=5, k=5, q=15, 
                                          batch_size=10, n_modes=2)
        elif ep_type == "domain_shift":
            episodes = create_episode_batch(generator, ep_type, N=5, k=5, q=15, 
                                          batch_size=10, shift_factor=1.5)
        else:
            episodes = create_episode_batch(generator, ep_type, N=5, k=5, q=15, batch_size=10)
        
        support_data, support_labels, query_data, query_labels = episodes[0]
        print(f"  Support shape: {support_data.shape}, Query shape: {query_data.shape}")
        print(f"  Support labels: {support_labels.unique()}")
        print(f"  Query labels: {query_labels.unique()}")
        print(f"  Support mean norm: {support_data.norm(dim=1).mean():.3f}")
        print(f"  Query mean norm: {query_data.norm(dim=1).mean():.3f}")
    
    print("\nPreprocessing module test completed successfully!")
