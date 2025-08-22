import torch
import torchvision
import torchvision.transforms as transforms
import os

def get_cifar10_data(batch_size, T, data_dir='./data'):
    """
    Downloads and prepares the CIFAR-10 dataset.
    Applies transformations and wraps the data loaders for rate encoding.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    class RateEncodingWrapper:
        """
        Wraps a DataLoader to apply rate encoding to images on-the-fly.
        """
        def __init__(self, loader, T):
            self.loader = loader
            self.T = T

        def __len__(self):
            return len(self.loader)

        def __iter__(self):
            for img, label in self.loader:
                # Repeat image T times for SNN simulation
                # Shape: [N, C, H, W] -> [T, N, C, H, W]
                img = img.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
                yield img, label
                
    return RateEncodingWrapper(train_loader, T), RateEncodingWrapper(test_loader, T)
