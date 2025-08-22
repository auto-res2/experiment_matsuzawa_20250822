import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import random


class PermutedMNIST(Dataset):
    """Permuted MNIST dataset for continual learning."""
    
    def __init__(self, root='./data', train=True, permutation=None, download=True):
        self.mnist = torchvision.datasets.MNIST(
            root=root, train=train, download=download,
            transform=transforms.ToTensor()
        )
        self.permutation = permutation
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        if self.permutation is not None:
            image = image.view(-1)[self.permutation].view(1, 28, 28)
        return image, label


class SplitCIFAR100(Dataset):
    """Split CIFAR-100 dataset for continual learning."""
    
    def __init__(self, root='./data', train=True, task_id=0, classes_per_task=5, download=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        self.cifar100 = torchvision.datasets.CIFAR100(
            root=root, train=train, download=download, transform=transform
        )
        
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task
        self.task_classes = list(range(start_class, end_class))
        
        self.indices = [i for i, (_, label) in enumerate(self.cifar100) 
                       if label in self.task_classes]
        
        self.label_map = {old_label: new_label for new_label, old_label 
                         in enumerate(self.task_classes)}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.cifar100[real_idx]
        new_label = self.label_map[label]
        return image, new_label


def create_continual_learning_datasets(dataset_name, num_tasks=10):
    """Create datasets for continual learning experiments."""
    
    if dataset_name == 'permuted_mnist':
        datasets = []
        permutations = []
        
        for task_id in range(num_tasks):
            if task_id == 0:
                perm = None
            else:
                perm = torch.randperm(784)
            
            permutations.append(perm)
            
            train_dataset = PermutedMNIST(train=True, permutation=perm)
            test_dataset = PermutedMNIST(train=False, permutation=perm)
            
            datasets.append({
                'train': train_dataset,
                'test': test_dataset,
                'task_id': task_id,
                'permutation': perm
            })
            
        return datasets
    
    elif dataset_name == 'split_cifar100':
        datasets = []
        classes_per_task = 100 // num_tasks
        
        for task_id in range(num_tasks):
            train_dataset = SplitCIFAR100(train=True, task_id=task_id, 
                                        classes_per_task=classes_per_task)
            test_dataset = SplitCIFAR100(train=False, task_id=task_id, 
                                       classes_per_task=classes_per_task)
            
            datasets.append({
                'train': train_dataset,
                'test': test_dataset,
                'task_id': task_id,
                'classes': list(range(task_id * classes_per_task, 
                                    (task_id + 1) * classes_per_task))
            })
            
        return datasets
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_data_loaders(dataset, batch_size=64, shuffle=True):
    """Create data loaders for training and testing."""
    
    train_loader = DataLoader(
        dataset['train'], 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset['test'], 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader
