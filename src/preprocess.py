import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math

class SyntheticPatternsDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=32, pattern_type='mixed'):
        self.num_samples = num_samples
        self.img_size = img_size
        self.pattern_type = pattern_type
        
        self.data, self.labels = self._generate_data()
    
    def _generate_data(self):
        data = []
        labels = []
        
        for i in range(self.num_samples):
            if self.pattern_type == 'mixed':
                pattern_idx = i % 4
            else:
                pattern_idx = {'vertical': 0, 'horizontal': 1, 'checkerboard': 2, 'center': 3}[self.pattern_type]
            
            img = self._create_pattern(pattern_idx)
            data.append(img)
            labels.append(pattern_idx)
        
        return torch.stack(data), torch.tensor(labels)
    
    def _create_pattern(self, pattern_idx):
        img = torch.zeros(3, self.img_size, self.img_size)
        
        if pattern_idx == 0:  # Vertical stripes
            for i in range(0, self.img_size, 4):
                img[:, :, i:i+2] = 1.0
        elif pattern_idx == 1:  # Horizontal stripes  
            for i in range(0, self.img_size, 4):
                img[:, i:i+2, :] = 1.0
        elif pattern_idx == 2:  # Checkerboard
            for i in range(0, self.img_size, 8):
                for j in range(0, self.img_size, 8):
                    if (i//8 + j//8) % 2 == 0:
                        img[:, i:i+8, j:j+8] = 1.0
        elif pattern_idx == 3:  # Center blob
            center = self.img_size // 2
            radius = self.img_size // 4
            y, x = torch.meshgrid(torch.arange(self.img_size), torch.arange(self.img_size), indexing='ij')
            mask = ((x - center)**2 + (y - center)**2) < radius**2
            img[:, mask] = 1.0
        
        img += torch.randn_like(img) * 0.1
        img = torch.clamp(img, 0, 1)
        
        return img
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_2d_sincos_pos_embed(embed_dim, grid_size, device):
    H, W = grid_size
    grid_h = np.arange(H, dtype=np.float32)
    grid_w = np.arange(W, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(2, 1, H, W)

    def _pe(pos, d_model):
        omega = np.arange(d_model // 2, dtype=np.float32)
        omega /= d_model / 2.
        omega = 1.0 / (10000**omega)
        out = pos.reshape(-1)[:, None] * omega[None]
        pe = np.concatenate([np.sin(out), np.cos(out)], axis=1)
        return pe

    assert embed_dim % 4 == 0, 'embed_dim should be divisible by 4'
    d_each = embed_dim // 2
    pe_h = _pe(grid[1], d_each)
    pe_w = _pe(grid[0], d_each)
    pe = np.concatenate([pe_h, pe_w], axis=1)
    pe = torch.tensor(pe, dtype=torch.float32, device=device).unsqueeze(0)
    return pe

def create_synthetic_datasets():
    print("Creating synthetic pattern datasets...")
    
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        num_samples = {'train': 800, 'val': 200, 'test': 200}[split]
        datasets[split] = SyntheticPatternsDataset(
            num_samples=num_samples,
            img_size=32,
            pattern_type='mixed'
        )
    
    print(f"Created datasets: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}")
    
    return datasets
