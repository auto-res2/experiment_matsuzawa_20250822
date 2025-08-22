import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def get_dataloader(batch_size=64, num_workers=4, data_dir='./data'):
    """
    Create CIFAR-10 dataloader with preprocessing for AOFM training.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        data_dir: Directory to store/load CIFAR-10 data
    
    Returns:
        DataLoader for CIFAR-10 training data
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True, 
        pin_memory=True
    )
    
    return dataloader

def get_test_dataloader(batch_size=64, num_workers=4, data_dir='./data'):
    """
    Create CIFAR-10 test dataloader for evaluation.
    
    Args:
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        data_dir: Directory to load CIFAR-10 data from
    
    Returns:
        DataLoader for CIFAR-10 test data
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return dataloader
