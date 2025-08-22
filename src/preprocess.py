import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_dataloaders(config):
    """Prepares CIFAR-100 dataloaders for pre-training and linear evaluation."""
    print("--- Preparing DataLoaders ---")
    # Augmentations for contrastive pre-training (MoCo v2 style)
    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Augmentations for linear evaluation
    linear_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    linear_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Pre-training dataset (returns two views)
    pretrain_dataset = torchvision.datasets.CIFAR100(
        root=config['DATA_DIR'], train=True, download=True, transform=TwoCropTransform(pretrain_transform))

    # Linear evaluation datasets (returns one view and label)
    linear_train_dataset = torchvision.datasets.CIFAR100(
        root=config['DATA_DIR'], train=True, download=True, transform=linear_train_transform)
    linear_test_dataset = torchvision.datasets.CIFAR100(
        root=config['DATA_DIR'], train=False, download=True, transform=linear_test_transform)
    
    if config['TEST_MODE']:
        pretrain_dataset = Subset(pretrain_dataset, range(config['PRETRAIN_BATCH_SIZE'] * 2))
        linear_train_dataset = Subset(linear_train_dataset, range(config['LINEAR_BATCH_SIZE'] * 2))
        linear_test_dataset = Subset(linear_test_dataset, range(config['LINEAR_BATCH_SIZE'] * 2))

    pretrain_loader = DataLoader(pretrain_dataset, batch_size=config['PRETRAIN_BATCH_SIZE'], shuffle=True,
                                 num_workers=config['NUM_WORKERS'], pin_memory=True, drop_last=True)
    
    linear_train_loader = DataLoader(linear_train_dataset, batch_size=config['LINEAR_BATCH_SIZE'], shuffle=True,
                                     num_workers=config['NUM_WORKERS'], pin_memory=True)
    
    linear_test_loader = DataLoader(linear_test_dataset, batch_size=config['LINEAR_BATCH_SIZE'], shuffle=False,
                                    num_workers=config['NUM_WORKERS'], pin_memory=True)

    print(f"Pre-train loader: {len(pretrain_loader)} batches")
    print(f"Linear train loader: {len(linear_train_loader)} batches")
    print(f"Linear test loader: {len(linear_test_loader)} batches")
    return pretrain_loader, linear_train_loader, linear_test_loader
