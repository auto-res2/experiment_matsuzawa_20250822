#!/usr/bin/env python

"""
Data preprocessing module for CAMoE-Diff experiment.
Creates synthetic datasets with varying complexity levels to test content-awareness.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, List


class SyntheticDatasetGenerator:
    """Generates synthetic datasets with varying complexity levels."""
    
    def __init__(self, image_size: int = 64, device: str = "cuda"):
        self.image_size = image_size
        self.device = device
    
    def create_simple_texture(self, batch_size: int) -> torch.Tensor:
        """Creates simple solid color textures."""
        images = []
        for i in range(batch_size):
            color = torch.rand(3, device=self.device)
            img = color.view(3, 1, 1).expand(3, self.image_size, self.image_size).clone()
            img += torch.randn_like(img) * 0.05
            images.append(img)
        return torch.stack(images)
    
    def create_geometric_patterns(self, batch_size: int) -> torch.Tensor:
        """Creates medium complexity geometric patterns."""
        images = []
        for i in range(batch_size):
            img = torch.zeros(3, self.image_size, self.image_size, device=self.device)
            
            center = self.image_size // 2
            radius = self.image_size // 4
            
            y, x = torch.meshgrid(
                torch.arange(self.image_size, device=self.device),
                torch.arange(self.image_size, device=self.device),
                indexing='ij'
            )
            circle_mask = ((x - center) ** 2 + (y - center) ** 2) < radius ** 2
            
            for c in range(3):
                img[c][circle_mask] = torch.rand(1, device=self.device)
            
            img += torch.randn_like(img) * 0.1
            images.append(img)
        return torch.stack(images)
    
    def create_complex_textures(self, batch_size: int) -> torch.Tensor:
        """Creates complex high-frequency textures."""
        images = []
        for i in range(batch_size):
            img = torch.randn(3, self.image_size, self.image_size, device=self.device)
            
            y, x = torch.meshgrid(
                torch.arange(self.image_size, device=self.device),
                torch.arange(self.image_size, device=self.device),
                indexing='ij'
            )
            
            freq1 = torch.rand(1, device=self.device) * 0.5 + 0.1
            freq2 = torch.rand(1, device=self.device) * 0.5 + 0.1
            
            pattern = torch.sin(x * freq1) * torch.cos(y * freq2)
            img[0] += pattern * 0.5
            img[1] += torch.sin(x * freq2) * torch.sin(y * freq1) * 0.5
            img[2] += torch.cos(x * freq1 + y * freq2) * 0.5
            
            images.append(img)
        return torch.stack(images)
    
    def generate_mixed_dataset(self, total_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a mixed dataset with varying complexity levels."""
        samples_per_type = total_samples // 3
        
        simple_data = self.create_simple_texture(samples_per_type)
        geometric_data = self.create_geometric_patterns(samples_per_type)
        complex_data = self.create_complex_textures(samples_per_type)
        
        all_data = torch.cat([simple_data, geometric_data, complex_data], dim=0)
        labels = torch.cat([
            torch.zeros(samples_per_type, dtype=torch.long),  # Simple
            torch.ones(samples_per_type, dtype=torch.long),   # Geometric
            torch.full((samples_per_type,), 2, dtype=torch.long)  # Complex
        ])
        
        perm = torch.randperm(total_samples)
        all_data = all_data[perm]
        labels = labels[perm]
        
        all_data = torch.clamp(all_data, -1, 1)
        
        return all_data, labels


def create_datasets(config: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates training and validation datasets."""
    generator = SyntheticDatasetGenerator(
        image_size=config['image_size'],
        device=config['device']
    )
    
    train_data, train_labels = generator.generate_mixed_dataset(config['train_samples'])
    
    val_data, val_labels = generator.generate_mixed_dataset(config['val_samples'])
    
    print(f"Generated training dataset: {train_data.shape}")
    print(f"Generated validation dataset: {val_data.shape}")
    print(f"Complexity distribution - Simple: {(train_labels == 0).sum()}, "
          f"Geometric: {(train_labels == 1).sum()}, Complex: {(train_labels == 2).sum()}")
    
    return train_data, train_labels, val_data, val_labels


def save_sample_images(data: torch.Tensor, labels: torch.Tensor, save_dir: str, num_samples: int = 9):
    """Saves sample images for visualization."""
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Sample Dataset Images by Complexity', fontsize=16)
    
    complexity_names = ['Simple', 'Geometric', 'Complex']
    
    for complexity in range(3):
        indices = torch.where(labels == complexity)[0][:3]
        for i, idx in enumerate(indices):
            img = data[idx].cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img + 1) / 2  # Denormalize from [-1,1] to [0,1]
            img = np.clip(img, 0, 1)
            
            axes[complexity, i].imshow(img)
            axes[complexity, i].set_title(f'{complexity_names[complexity]} #{i+1}')
            axes[complexity, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_dataset.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sample images saved to {save_dir}/sample_dataset.pdf")


if __name__ == "__main__":
    config = {
        'image_size': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'train_samples': 120,
        'val_samples': 60
    }
    
    train_data, train_labels, val_data, val_labels = create_datasets(config)
    save_sample_images(train_data, train_labels, '.research/iteration1/images/')
