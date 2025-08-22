"""
AccuTune Preprocessing Module
Handles data preprocessing for AccuTune experiments including CIFAR-10, synthetic data, and text data.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    import torchvision
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class SyntheticDataset(Dataset):
    """Synthetic dataset for quick testing"""
    def __init__(self, num_samples=1000, input_dim=3072, num_classes=10, noise_std=0.1):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        torch.manual_seed(42)
        self.data = torch.randn(num_samples, input_dim)
        
        weights = torch.randn(input_dim, num_classes) * 0.1
        logits = self.data @ weights + torch.randn(num_samples, num_classes) * noise_std
        self.labels = logits.argmax(dim=1)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TextDataset(Dataset):
    """Simple text dataset for transformer experiments"""
    def __init__(self, num_samples=1000, seq_len=128, vocab_size=8192):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        torch.manual_seed(42)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]


def get_cifar10_loaders(batch_size=32, num_workers=2, quick_test=False):
    """Get CIFAR-10 data loaders"""
    if not HAS_TORCHVISION:
        print("torchvision not available, using synthetic data")
        return get_synthetic_loaders(batch_size, input_shape=(3, 32, 32), quick_test=quick_test)
    
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    try:
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        if quick_test:
            train_indices = torch.randperm(len(train_dataset))[:1000]
            test_indices = torch.randperm(len(test_dataset))[:200]
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"Failed to load CIFAR-10: {e}, using synthetic data")
        return get_synthetic_loaders(batch_size, input_shape=(3, 32, 32), quick_test=quick_test)


def get_synthetic_loaders(batch_size=32, input_shape=(3, 32, 32), num_classes=10, quick_test=False):
    """Get synthetic data loaders"""
    input_dim = np.prod(input_shape)
    
    train_size = 1000 if quick_test else 5000
    test_size = 200 if quick_test else 1000
    
    train_dataset = SyntheticDataset(train_size, input_dim, num_classes)
    test_dataset = SyntheticDataset(test_size, input_dim, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_text_loaders(batch_size=16, seq_len=128, vocab_size=8192, quick_test=False):
    """Get text data loaders for transformer experiments"""
    train_size = 500 if quick_test else 2000
    test_size = 100 if quick_test else 500
    
    train_dataset = TextDataset(train_size, seq_len, vocab_size)
    test_dataset = TextDataset(test_size, seq_len, vocab_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def preprocess_data(experiment_type="mlp", quick_test=False, batch_size=32):
    """Main preprocessing function"""
    print(f"Preprocessing data for {experiment_type} experiment (quick_test={quick_test})")
    
    if experiment_type == "mlp":
        return get_cifar10_loaders(batch_size=batch_size, quick_test=quick_test)
    elif experiment_type == "cnn":
        return get_cifar10_loaders(batch_size=batch_size, quick_test=quick_test)
    elif experiment_type == "transformer":
        return get_text_loaders(batch_size=batch_size, quick_test=quick_test)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


if __name__ == "__main__":
    print("Testing data preprocessing...")
    
    for exp_type in ["mlp", "cnn", "transformer"]:
        print(f"\nTesting {exp_type}:")
        train_loader, test_loader = preprocess_data(exp_type, quick_test=True, batch_size=8)
        
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))
        
        print(f"  Train batch shapes: {[x.shape for x in train_batch]}")
        print(f"  Test batch shapes: {[x.shape for x in test_batch]}")
        print(f"  Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    print("\nPreprocessing test completed successfully!")
