#!/usr/bin/env python3
"""
Data preprocessing and synthetic data generation for Rev-SS2D experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class SyntheticDataGenerator:
    """Generate synthetic datasets for Rev-SS2D experiments"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def generate_classification_data(
        self, 
        batch_size: int, 
        channels: int, 
        height: int, 
        width: int, 
        num_classes: int,
        noise_level: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Generate synthetic classification data with spatial patterns"""
        
        images = torch.randn(batch_size, channels, height, width, device=self.device)
        labels = torch.randint(0, num_classes, (batch_size,), device=self.device)
        
        for i in range(batch_size):
            class_id = labels[i].item()
            
            if class_id % 4 == 0:  # Horizontal stripes
                stripe_freq = 2 + class_id // 4
                y_coords = torch.arange(height, device=self.device).float()
                pattern = torch.sin(2 * np.pi * stripe_freq * y_coords / height)
                images[i] += 0.5 * pattern.view(1, height, 1).expand(channels, height, width)
                
            elif class_id % 4 == 1:  # Vertical stripes  
                stripe_freq = 2 + class_id // 4
                x_coords = torch.arange(width, device=self.device).float()
                pattern = torch.sin(2 * np.pi * stripe_freq * x_coords / width)
                images[i] += 0.5 * pattern.view(1, 1, width).expand(channels, height, width)
                
            elif class_id % 4 == 2:  # Checkerboard
                y_coords = torch.arange(height, device=self.device).float()
                x_coords = torch.arange(width, device=self.device).float()
                Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
                freq = 3 + class_id // 4
                pattern = torch.sin(2 * np.pi * freq * Y / height) * torch.sin(2 * np.pi * freq * X / width)
                images[i] += 0.5 * pattern.unsqueeze(0).expand(channels, height, width)
                
            else:  # Radial pattern
                center_y, center_x = height // 2, width // 2
                y_coords = torch.arange(height, device=self.device).float() - center_y
                x_coords = torch.arange(width, device=self.device).float() - center_x
                Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
                radius = torch.sqrt(Y**2 + X**2)
                freq = 2 + class_id // 4
                pattern = torch.sin(2 * np.pi * freq * radius / min(height, width))
                images[i] += 0.5 * pattern.unsqueeze(0).expand(channels, height, width)
        
        images += noise_level * torch.randn_like(images)
        
        images = torch.tanh(images)
        
        return {
            'images': images,
            'labels': labels,
            'metadata': {
                'batch_size': batch_size,
                'channels': channels,
                'height': height,
                'width': width,
                'num_classes': num_classes,
                'noise_level': noise_level
            }
        }
    
    def generate_segmentation_data(
        self,
        batch_size: int,
        channels: int, 
        height: int,
        width: int,
        num_classes: int
    ) -> Dict[str, torch.Tensor]:
        """Generate synthetic segmentation data with geometric shapes"""
        
        images = torch.randn(batch_size, channels, height, width, device=self.device)
        masks = torch.zeros(batch_size, height, width, dtype=torch.long, device=self.device)
        
        for i in range(batch_size):
            num_shapes = np.random.randint(2, 6)
            
            for _ in range(num_shapes):
                shape_class = np.random.randint(1, num_classes)
                
                center_y = np.random.randint(height // 4, 3 * height // 4)
                center_x = np.random.randint(width // 4, 3 * width // 4)
                size = np.random.randint(min(height, width) // 8, min(height, width) // 4)
                
                y_coords = torch.arange(height, device=self.device).float()
                x_coords = torch.arange(width, device=self.device).float()
                Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
                
                if shape_class % 3 == 1:  # Circle
                    dist = torch.sqrt((Y - center_y)**2 + (X - center_x)**2)
                    shape_mask = dist <= size
                elif shape_class % 3 == 2:  # Rectangle
                    shape_mask = (torch.abs(Y - center_y) <= size) & (torch.abs(X - center_x) <= size)
                else:  # Diamond
                    shape_mask = (torch.abs(Y - center_y) + torch.abs(X - center_x)) <= size
                
                masks[i][shape_mask] = shape_class
                
                pattern_intensity = 0.8 + 0.4 * np.random.randn()
                for c in range(channels):
                    images[i, c][shape_mask] += pattern_intensity * (shape_class / num_classes)
        
        images = torch.tanh(images)
        
        return {
            'images': images,
            'masks': masks,
            'metadata': {
                'batch_size': batch_size,
                'channels': channels,
                'height': height,
                'width': width,
                'num_classes': num_classes
            }
        }
    
    def create_memory_stress_data(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int
    ) -> torch.Tensor:
        """Create data specifically for memory stress testing"""
        
        images = torch.randn(batch_size, channels, height, width, device=self.device)
        
        for i in range(batch_size):
            y_coords = torch.arange(height, device=self.device).float()
            x_coords = torch.arange(width, device=self.device).float()
            Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            freq_y = np.random.randint(8, 16)
            freq_x = np.random.randint(8, 16)
            
            pattern = torch.sin(2 * np.pi * freq_y * Y / height) * torch.sin(2 * np.pi * freq_x * X / width)
            images[i] += 0.3 * pattern.unsqueeze(0).expand(channels, height, width)
            
            noise = torch.randn(channels, height, width, device=self.device)
            images[i] += 0.2 * noise
        
        return images


class DataPreprocessor:
    """Preprocessing utilities for real datasets"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def normalize_images(self, images: torch.Tensor, method: str = 'imagenet') -> torch.Tensor:
        """Normalize images using specified method"""
        
        if method == 'imagenet':
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            return (images - mean) / std
        elif method == 'zero_one':
            return (images - images.min()) / (images.max() - images.min())
        elif method == 'standard':
            return (images - images.mean()) / images.std()
        else:
            return images
    
    def augment_images(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation"""
        
        batch_size = images.shape[0]
        augmented_images = images.clone()
        
        for i in range(batch_size):
            if torch.rand(1) > 0.5:
                augmented_images[i] = torch.flip(augmented_images[i], dims=[2])
            
            if torch.rand(1) > 0.5:
                augmented_images[i] = torch.flip(augmented_images[i], dims=[1])
            
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                augmented_images[i] = torch.rot90(augmented_images[i], k=k, dims=[1, 2])
        
        return augmented_images, labels
    
    def create_tiles(self, images: torch.Tensor, tile_size: int, overlap: int = 0) -> torch.Tensor:
        """Split images into tiles for streaming processing"""
        
        B, C, H, W = images.shape
        stride = tile_size - overlap
        
        n_tiles_h = (H - overlap) // stride
        n_tiles_w = (W - overlap) // stride
        
        tiles = []
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                start_h = i * stride
                end_h = start_h + tile_size
                start_w = j * stride  
                end_w = start_w + tile_size
                
                if end_h <= H and end_w <= W:
                    tile = images[:, :, start_h:end_h, start_w:end_w]
                    tiles.append(tile)
        
        if tiles:
            return torch.stack(tiles, dim=1)  # (B, num_tiles, C, tile_size, tile_size)
        else:
            return images.unsqueeze(1)  # (B, 1, C, H, W)
