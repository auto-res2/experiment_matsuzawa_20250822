import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_cifar100_loaders(batch_size=128, test_run=False, data_dir='./data'):
    """Prepares CIFAR-100 DataLoaders with proper preprocessing."""
    
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = datasets.CIFAR100(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    if test_run:
        subset_size = min(batch_size * 4, len(train_dataset))
        train_indices = torch.randperm(len(train_dataset))[:subset_size]
        test_indices = torch.randperm(len(test_dataset))[:subset_size//4]
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    return train_loader, test_loader

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    print("Testing CIFAR-100 data loading...")
    set_seed(42)
    
    train_loader, test_loader = get_cifar100_loaders(batch_size=64, test_run=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"Target range: [{target.min()}, {target.max()}]")
        break
    
    print("Data loading test completed successfully!")
