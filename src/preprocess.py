import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, Dataset, Subset
import random


class TemporalDirichletSampler:
    """Simulates temporal distribution shift with Dirichlet-skewed class distributions."""
    
    def __init__(self, dataset, num_classes: int = 10, batch_size: int = 64, 
                 alpha_base: float = 1.0, alpha_skew: float = 0.1, 
                 shift_frequency: int = 100):
        self.dataset = dataset
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.alpha_base = alpha_base
        self.alpha_skew = alpha_skew
        self.shift_frequency = shift_frequency
        
        self.class_indices = {i: [] for i in range(num_classes)}
        for idx in range(len(dataset)):
            try:
                item = dataset[idx]
                if len(item) == 2:
                    _, label = item
                elif len(item) == 3:
                    _, _, label = item
                else:
                    continue
                if isinstance(label, torch.Tensor):
                    label = label.item()
                self.class_indices[label].append(idx)
            except Exception:
                continue
        
        self.batch_count = 0
        self.current_probs = np.ones(num_classes) / num_classes
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.batch_count % self.shift_frequency == 0:
            alphas = np.full(self.num_classes, self.alpha_base)
            skew_classes = np.random.choice(self.num_classes, 
                                          size=np.random.randint(1, 4), 
                                          replace=False)
            alphas[skew_classes] += self.alpha_skew
            self.current_probs = np.random.dirichlet(alphas)
        
        batch_indices = []
        target_counts = np.random.multinomial(self.batch_size, self.current_probs)
        
        for class_id, count in enumerate(target_counts):
            if count > 0 and len(self.class_indices[class_id]) > 0:
                selected = np.random.choice(self.class_indices[class_id], 
                                          size=min(count, len(self.class_indices[class_id])), 
                                          replace=count > len(self.class_indices[class_id]))
                batch_indices.extend(selected)
        
        while len(batch_indices) < self.batch_size:
            class_id = np.random.choice(self.num_classes)
            if self.class_indices[class_id]:
                batch_indices.append(np.random.choice(self.class_indices[class_id]))
        
        self.batch_count += 1
        return batch_indices[:self.batch_size]


def get_cifar10_transforms():
    """Get CIFAR-10 transforms for source and target domains."""
    
    source_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    strong_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return source_transform, target_transform, strong_transform


def create_synthetic_corruption(images: torch.Tensor, corruption_type: str = "gaussian_noise", 
                              severity: float = 0.1) -> torch.Tensor:
    """Apply synthetic corruption to images."""
    corrupted = images.clone()
    
    if corruption_type == "gaussian_noise":
        noise = torch.randn_like(images) * severity
        corrupted = torch.clamp(images + noise, 0, 1)
    elif corruption_type == "brightness":
        corrupted = torch.clamp(images + severity, 0, 1)
    elif corruption_type == "contrast":
        mean = images.mean(dim=(2, 3), keepdim=True)
        corrupted = torch.clamp(mean + (images - mean) * (1 + severity), 0, 1)
    
    return corrupted


class CorruptedDataset(Dataset):
    """Dataset wrapper that applies corruption on-the-fly."""
    
    def __init__(self, base_dataset, corruption_type: str = "gaussian_noise", 
                 severity: float = 0.1, strong_aug_transform=None):
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity
        self.strong_aug_transform = strong_aug_transform
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        if isinstance(image, torch.Tensor):
            corrupted = create_synthetic_corruption(image.unsqueeze(0), 
                                                  self.corruption_type, 
                                                  self.severity).squeeze(0)
        else:
            to_tensor = transforms.ToTensor()
            image_tensor = to_tensor(image)
            corrupted = create_synthetic_corruption(image_tensor.unsqueeze(0), 
                                                  self.corruption_type, 
                                                  self.severity).squeeze(0)
        
        if self.strong_aug_transform:
            to_pil = transforms.ToPILImage()
            strong_aug = self.strong_aug_transform(to_pil(corrupted))
            return corrupted, strong_aug, label
        
        return corrupted, label


def load_datasets(data_dir: str = "./data", batch_size: int = 64, 
                 corruption_type: str = "gaussian_noise", severity: float = 0.1):
    """Load CIFAR-10 datasets with corruption simulation."""
    
    os.makedirs(data_dir, exist_ok=True)
    
    source_transform, target_transform, strong_transform = get_cifar10_transforms()
    
    try:
        source_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=source_transform
        )
    except:
        source_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=source_transform
        )
    
    corrupted_dataset = CorruptedDataset(
        source_dataset, corruption_type=corruption_type, 
        severity=severity, strong_aug_transform=strong_transform
    )
    
    temporal_sampler = TemporalDirichletSampler(
        corrupted_dataset, num_classes=10, batch_size=batch_size,
        alpha_base=1.0, alpha_skew=0.1, shift_frequency=50
    )
    
    return source_dataset, corrupted_dataset, temporal_sampler


def create_streaming_dataloader(dataset, sampler, batch_size: int = 64, num_workers: int = 2):
    """Create a streaming dataloader with temporal distribution shifts."""
    
    class StreamingDataLoader:
        def __init__(self, dataset, sampler, batch_size):
            self.dataset = dataset
            self.sampler = sampler
            self.batch_size = batch_size
            
        def __iter__(self):
            return self
            
        def __next__(self):
            try:
                indices = next(self.sampler)
                batch_data = []
                for idx in indices:
                    batch_data.append(self.dataset[idx])
                
                if len(batch_data[0]) == 3:  # With strong augmentation
                    images = torch.stack([item[0] for item in batch_data])
                    strong_augs = torch.stack([item[1] for item in batch_data])
                    labels = torch.tensor([item[2] for item in batch_data])
                    return images, strong_augs, labels
                else:
                    images = torch.stack([item[0] for item in batch_data])
                    labels = torch.tensor([item[1] for item in batch_data])
                    return images, labels
                    
            except StopIteration:
                raise StopIteration
    
    return StreamingDataLoader(dataset, sampler, batch_size)


if __name__ == "__main__":
    print("Testing data preprocessing...")
    
    source_dataset, corrupted_dataset, temporal_sampler = load_datasets(
        batch_size=32, corruption_type="gaussian_noise", severity=0.1
    )
    
    streaming_loader = create_streaming_dataloader(
        corrupted_dataset, temporal_sampler, batch_size=32
    )
    
    for i, batch in enumerate(streaming_loader):
        if i >= 3:  # Test first 3 batches
            break
        
        if len(batch) == 3:
            images, strong_augs, labels = batch
            print(f"Batch {i}: Images {images.shape}, Strong augs {strong_augs.shape}, Labels {labels.shape}")
        else:
            images, labels = batch
            print(f"Batch {i}: Images {images.shape}, Labels {labels.shape}")
        
        unique, counts = torch.unique(labels, return_counts=True)
        print(f"  Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    print("Data preprocessing test completed successfully!")
