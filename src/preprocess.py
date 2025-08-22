"""
Data preprocessing for SEEDS experiments
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.seeds_config import config

class SyntheticDataset(Dataset):
    """Synthetic discrete data for testing SEEDS"""
    
    def __init__(self, n_samples: int, shape: Tuple[int, ...], K: int, pattern: str = "checkerboard"):
        self.n_samples = n_samples
        self.shape = shape
        self.K = K
        self.pattern = pattern
        self.data = self._generate_data()
    
    def _generate_data(self) -> torch.Tensor:
        if self.pattern == "checkerboard":
            return self._generate_checkerboard()
        elif self.pattern == "stripes":
            return self._generate_stripes()
        else:
            return self._generate_random()
    
    def _generate_checkerboard(self) -> torch.Tensor:
        """Generate checkerboard patterns"""
        data = torch.zeros((self.n_samples,) + self.shape, dtype=torch.long)
        for i in range(self.n_samples):
            if len(self.shape) == 2:  # 2D image-like
                H, W = self.shape
                for h in range(H):
                    for w in range(W):
                        data[i, h, w] = ((h + w) % 2) * (self.K - 1)
            elif len(self.shape) == 1:  # 1D sequence-like
                L = self.shape[0]
                for l in range(L):
                    data[i, l] = (l % 2) * (self.K - 1)
        return data
    
    def _generate_stripes(self) -> torch.Tensor:
        """Generate stripe patterns"""
        data = torch.zeros((self.n_samples,) + self.shape, dtype=torch.long)
        for i in range(self.n_samples):
            if len(self.shape) == 2:  # 2D
                H, W = self.shape
                stripe_width = max(1, H // 4)
                for h in range(H):
                    val = (h // stripe_width) % self.K
                    data[i, h, :] = val
            elif len(self.shape) == 1:  # 1D
                L = self.shape[0]
                stripe_width = max(1, L // 4)
                for l in range(L):
                    data[i, l] = (l // stripe_width) % self.K
        return data
    
    def _generate_random(self) -> torch.Tensor:
        """Generate random patterns"""
        return torch.randint(0, self.K, (self.n_samples,) + self.shape)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_synthetic_datasets():
    """Create synthetic datasets for all experiments"""
    print("Creating synthetic datasets...")
    
    image_train = SyntheticDataset(
        n_samples=config.n_samples,
        shape=(config.image_size, config.image_size),
        K=config.K,
        pattern="checkerboard"
    )
    
    image_test = SyntheticDataset(
        n_samples=config.n_test_samples,
        shape=(config.image_size, config.image_size),
        K=config.K,
        pattern="checkerboard"
    )
    
    seq_train = SyntheticDataset(
        n_samples=config.n_samples,
        shape=(config.seq_length,),
        K=config.K,
        pattern="stripes"
    )
    
    seq_test = SyntheticDataset(
        n_samples=config.n_test_samples,
        shape=(config.seq_length,),
        K=config.K,
        pattern="stripes"
    )
    
    os.makedirs(config.data_dir, exist_ok=True)
    
    torch.save(image_train, os.path.join(config.data_dir, "image_train.pt"))
    torch.save(image_test, os.path.join(config.data_dir, "image_test.pt"))
    torch.save(seq_train, os.path.join(config.data_dir, "seq_train.pt"))
    torch.save(seq_test, os.path.join(config.data_dir, "seq_test.pt"))
    
    print(f"Saved datasets to {config.data_dir}")
    print(f"Image datasets: {len(image_train)} train, {len(image_test)} test")
    print(f"Sequence datasets: {len(seq_train)} train, {len(seq_test)} test")
    
    return {
        'image_train': image_train,
        'image_test': image_test,
        'seq_train': seq_train,
        'seq_test': seq_test
    }

def load_datasets():
    """Load existing datasets"""
    datasets = {}
    
    try:
        datasets['image_train'] = torch.load(os.path.join(config.data_dir, "image_train.pt"), weights_only=False)
        datasets['image_test'] = torch.load(os.path.join(config.data_dir, "image_test.pt"), weights_only=False)
        datasets['seq_train'] = torch.load(os.path.join(config.data_dir, "seq_train.pt"), weights_only=False)
        datasets['seq_test'] = torch.load(os.path.join(config.data_dir, "seq_test.pt"), weights_only=False)
        print("Loaded existing datasets")
    except FileNotFoundError:
        print("Datasets not found, creating new ones...")
        datasets = create_synthetic_datasets()
    
    return datasets

def create_dataloaders(datasets):
    """Create DataLoaders from datasets"""
    loaders = {}
    
    for name, dataset in datasets.items():
        shuffle = 'train' in name
        loaders[name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if config.device == "cuda" else False
        )
    
    return loaders

def main():
    """Main preprocessing function"""
    datasets = create_synthetic_datasets()
    loaders = create_dataloaders(datasets)
    print("Preprocessing completed successfully!")
    return datasets

if __name__ == "__main__":
    main()
