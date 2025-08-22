#!/usr/bin/env python3
"""
Data preprocessing for DASH-HiLo-Anchor experiments.
Creates synthetic datasets for quick testing and evaluation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class SyntheticObjectsDataset(Dataset):
    """
    Multi-pattern synthetic dataset to probe small-object sensitivity and robustness.
    Classes: 0=circle, 1=square, 2=triangle, 3=noisy blob.
    """
    def __init__(self, n_samples=1024, image_size=96, pattern='default', shrink_r=None, seed=0):
        super().__init__()
        self.n = n_samples
        self.S = image_size
        self.pattern = pattern
        self.shrink_r = shrink_r
        self.rng = np.random.RandomState(seed)
        self.num_classes = 4
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return self.n

    def _draw_shape(self, img, label):
        S = self.S
        sz_bins = [(8, 16), (16, 24), (24, 32)]
        bin_idx = self.rng.choice(3, p=[0.5, 0.35, 0.15])  # bias to small
        smin, smax = sz_bins[bin_idx]
        size = self.rng.randint(smin, smax)
        cx = self.rng.randint(size, S - size)
        cy = self.rng.randint(size, S - size)

        base_colors = {
            0: np.array([0.9, 0.1, 0.1]),  # red circle
            1: np.array([0.1, 0.9, 0.1]),  # green square
            2: np.array([0.1, 0.1, 0.9]),  # blue triangle
            3: np.array([0.7, 0.7, 0.1])   # yellowish blob
        }
        color = base_colors[int(label)].copy()
        if self.pattern == 'shifted_colors':
            color = np.clip(color + self.rng.uniform(-0.2, 0.2, size=3), 0.0, 1.0)

        Y, X = np.ogrid[:S, :S]
        if label == 0:
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= (size // 2) ** 2
        elif label == 1:
            mask = (np.abs(X - cx) <= size // 2) & (np.abs(Y - cy) <= size // 2)
        elif label == 2:
            x0, y0 = cx - size // 2, cy - size // 2
            mask = (X >= x0) & (X <= x0 + size) & (Y >= y0) & (Y <= y0 + (X - x0))
        else:
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= (size // 2) ** 2
            color = np.clip(color + self.rng.normal(0, 0.1, size=3), 0.0, 1.0)

        for c in range(3):
            img[c][mask] = color[c]
        return img

    def _add_background(self, img):
        if self.pattern == 'noisy_bg':
            img += self.rng.normal(0, 0.05, size=img.shape).astype(np.float32)
        else:
            img += self.rng.uniform(0, 0.02, size=img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        return img

    def _shrinkpad(self, img):
        if self.shrink_r is None or self.shrink_r >= 0.999:
            return img
        r = self.shrink_r
        S = self.S
        newS = max(1, int(S * r))
        pil = Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
        pil_small = pil.resize((newS, newS), resample=Image.Resampling.BICUBIC)
        canvas = np.tile(self.mean[None, None, :], (S, S, 1))
        y0 = (S - newS) // 2
        x0 = (S - newS) // 2
        canvas[y0:y0 + newS, x0:x0 + newS, :] = np.array(pil_small) / 255.0
        return canvas.transpose(2, 0, 1)

    def __getitem__(self, idx):
        S = self.S
        img = np.zeros((3, S, S), dtype=np.float32)
        label = self.rng.randint(0, 4)
        img = self._add_background(img)
        img = self._draw_shape(img, label)
        img = np.clip(img, 0.0, 1.0)
        img = self._shrinkpad(img)
        img = (img - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


def create_datasets(image_size=96, batch_size=32, num_workers=0):
    """Create train, validation, and test datasets."""
    
    train_dataset = SyntheticObjectsDataset(
        n_samples=2048, image_size=image_size, pattern='default', seed=42
    )
    
    val_dataset = SyntheticObjectsDataset(
        n_samples=512, image_size=image_size, pattern='shifted_colors', seed=123
    )
    
    test_dataset = SyntheticObjectsDataset(
        n_samples=512, image_size=image_size, pattern='noisy_bg', seed=456
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_shrinkpad_datasets(image_size=96, batch_size=32, shrink_ratios=None):
    """Create datasets with different shrink ratios for small-object sensitivity testing."""
    
    if shrink_ratios is None:
        shrink_ratios = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    datasets = {}
    loaders = {}
    
    for ratio in shrink_ratios:
        dataset = SyntheticObjectsDataset(
            n_samples=256, image_size=image_size, 
            pattern='default', shrink_r=ratio, seed=789
        )
        
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        datasets[f'shrink_{ratio}'] = dataset
        loaders[f'shrink_{ratio}'] = loader
    
    return datasets, loaders


def visualize_dataset_samples(dataset, save_path=None, num_samples=8):
    """Visualize sample images from the dataset."""
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    class_names = ['Circle', 'Square', 'Triangle', 'Blob']
    
    for i in range(num_samples):
        img, label = dataset[i]
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_vis = img.numpy()
        img_vis = img_vis * std[:, None, None] + mean[:, None, None]
        img_vis = np.clip(img_vis, 0, 1)
        img_vis = img_vis.transpose(1, 2, 0)
        
        axes[i].imshow(img_vis)
        axes[i].set_title(f'{class_names[label.item()]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset samples saved to: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    print("Testing data preprocessing...")
    
    train_loader, val_loader, test_loader = create_datasets()
    
    print(f"Train dataset: {len(train_loader.dataset)} samples")
    print(f"Val dataset: {len(val_loader.dataset)} samples") 
    print(f"Test dataset: {len(test_loader.dataset)} samples")
    
    os.makedirs('.research/iteration1/images', exist_ok=True)
    visualize_dataset_samples(
        train_loader.dataset, 
        save_path='.research/iteration1/images/dataset_samples.pdf'
    )
    
    print("Preprocessing test complete!")
