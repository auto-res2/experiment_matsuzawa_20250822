import math
import time
import random
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SyntheticShapesClassification(Dataset):
    """Simple 10-class synthetic classification dataset.
    - Image: 32x32 RGB
    - Label: quadrant/shape/color combination to create meaningful patterns
    - Two patterns: clean (sharp shapes) and corrupted (Gaussian noise/blur)
    """
    def __init__(self, n: int = 1000, img_size: int = 32, n_classes: int = 10, pattern: str = 'clean'):
        self.n = n
        self.img_size = img_size
        self.n_classes = n_classes
        self.pattern = pattern
        self.images, self.labels = self._generate()

    def _generate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        H = W = self.img_size
        imgs = torch.zeros((self.n, 3, H, W), dtype=torch.float32)
        labels = torch.zeros((self.n,), dtype=torch.long)
        for i in range(self.n):
            img = torch.zeros(3, H, W)
            c = torch.randint(low=0, high=self.n_classes, size=(1,)).item()
            shape = 'square' if c % 2 == 0 else 'circle'
            color_channel = c % 3
            pos_id = (c // 3) % 3  # 0,1,2
            size = H // 4
            if pos_id == 0:
                cx, cy = H // 4, W // 4
            elif pos_id == 1:
                cx, cy = H // 2, W // 2
            else:
                cx, cy = 3 * H // 4, 3 * W // 4
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            if shape == 'square':
                mask = (x >= cx - size) & (x <= cx + size) & (y >= cy - size) & (y <= cy + size)
            else:
                mask = ((x - cx) ** 2 + (y - cy) ** 2) <= (size ** 2)
            img[color_channel, mask] = 1.0
            img += 0.05 * torch.randn_like(img)
            img = torch.clamp(img, 0.0, 1.0)
            if self.pattern == 'corrupt':
                img = F.avg_pool2d(img.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
                img += 0.15 * torch.randn_like(img)
                img = torch.clamp(img, 0.0, 1.0)
            imgs[i] = img
            labels[i] = c
        return imgs, labels

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class SyntheticSegmentation(Dataset):
    """Toy segmentation dataset (3 classes) on 32x32 images with shapes.
    pattern='clean' or 'corrupt' to add noise/blur."""
    def __init__(self, n: int = 200, img_size: int = 32, n_classes: int = 3, pattern: str = 'clean'):
        self.n = n
        self.img_size = img_size
        self.n_classes = n_classes
        self.pattern = pattern
        self.images, self.labels = self._generate()

    def _generate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        H = W = self.img_size
        imgs = torch.zeros((self.n, 3, H, W), dtype=torch.float32)
        labels = torch.zeros((self.n, H, W), dtype=torch.long)
        for i in range(self.n):
            img = torch.zeros(3, H, W)
            lab = torch.zeros(H, W, dtype=torch.long)
            size = H // 5
            cx1, cy1 = H // 3, W // 3
            cx2, cy2 = 2 * H // 3, 2 * W // 3
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            rect = (x >= cx1 - size) & (x <= cx1 + size) & (y >= cy1 - size) & (y <= cy1 + size)
            circ = ((x - cx2) ** 2 + (y - cy2) ** 2) <= (size ** 2)
            lab[rect] = 1
            lab[circ] = 2
            img[0, rect] = 0.8
            img[1, circ] = 0.8
            img += 0.05 * torch.randn_like(img)
            img = torch.clamp(img, 0.0, 1.0)
            if self.pattern == 'corrupt':
                img = F.avg_pool2d(img.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
                img += 0.15 * torch.randn_like(img)
                img = torch.clamp(img, 0.0, 1.0)
            imgs[i] = img
            labels[i] = lab
        return imgs, labels

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def create_datasets():
    """Create synthetic datasets for classification and segmentation experiments."""
    print("Creating synthetic datasets...")
    
    train_cls_clean = SyntheticShapesClassification(n=800, pattern='clean')
    val_cls_clean = SyntheticShapesClassification(n=200, pattern='clean')
    test_cls_corrupt = SyntheticShapesClassification(n=200, pattern='corrupt')
    
    train_seg_clean = SyntheticSegmentation(n=160, pattern='clean')
    val_seg_clean = SyntheticSegmentation(n=40, pattern='clean')
    test_seg_corrupt = SyntheticSegmentation(n=40, pattern='corrupt')
    
    print(f"Classification: Train={len(train_cls_clean)}, Val={len(val_cls_clean)}, Test={len(test_cls_corrupt)}")
    print(f"Segmentation: Train={len(train_seg_clean)}, Val={len(val_seg_clean)}, Test={len(test_seg_corrupt)}")
    
    return {
        'cls_train': train_cls_clean,
        'cls_val': val_cls_clean,
        'cls_test': test_cls_corrupt,
        'seg_train': train_seg_clean,
        'seg_val': val_seg_clean,
        'seg_test': test_seg_corrupt
    }

if __name__ == "__main__":
    set_seed(42)
    datasets = create_datasets()
    print("Preprocessing completed successfully!")
