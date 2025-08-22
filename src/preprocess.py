import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

# =============================================================================
# 1. Backdoor Attack Implementations
# =============================================================================

class BackdoorAttack:
    """Base class for backdoor attacks."""
    def __init__(self, target_class=0):
        self.target_class = target_class

    def apply_trigger(self, image):
        raise NotImplementedError

class BadNetsAttack(BackdoorAttack):
    """Applies a 4x4 white square at the bottom-right corner."""
    def apply_trigger(self, image):
        c, h, w = image.shape
        image[:, h-5:h-1, w-5:w-1] = 1.0  # White square
        return image

class BlendedAttack(BackdoorAttack):
    """Blends a noise pattern into the image."""
    def __init__(self, target_class=0, alpha=0.15):
        super().__init__(target_class)
        self.alpha = alpha
        self.noise_pattern = torch.randn(3, 32, 32) * 0.2

    def apply_trigger(self, image):
        c, h, w = image.shape
        if self.noise_pattern.shape != image.shape:
            noise = transforms.functional.resize(self.noise_pattern, (h, w))
        else:
            noise = self.noise_pattern.to(image.device)
        triggered_image = (1 - self.alpha) * image + self.alpha * noise
        return torch.clamp(triggered_image, 0, 1)

class WaNetAttack(BackdoorAttack):
    """Applies a smooth warping field as a trigger."""
    def __init__(self, target_class=0, grid_rescale=1):
        super().__init__(target_class)
        self.grid_rescale = grid_rescale
        self.ins = torch.rand(1, 2, 4, 4) * 2 - 1
        self.ins = self.ins / torch.mean(torch.abs(self.ins))

    def apply_trigger(self, image):
        c, h, w = image.shape
        grid = self.ins.clone().detach().to(image.device)
        grid = F.interpolate(grid, size=(h, w), mode='bicubic', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)
        grid = grid.repeat(image.unsqueeze(0).shape[0], 1, 1, 1)
        
        identity_grid = F.affine_grid(torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32, device=image.device), (1, c, h, w))
        flow_grid = identity_grid + grid * self.grid_rescale
        
        triggered_image = F.grid_sample(image.unsqueeze(0), flow_grid, align_corners=True).squeeze(0)
        return triggered_image

class SinusoidalAttack(BackdoorAttack):
    """Adds a sinusoidal signal to the image (Clean-Label variant)."""
    def __init__(self, target_class=0, frequency=10, amplitude=0.1):
        super().__init__(target_class)
        self.frequency = frequency
        self.amplitude = amplitude

    def apply_trigger(self, image):
        c, h, w = image.shape
        signal = torch.zeros_like(image)
        for i in range(h):
            for j in range(w):
                signal[:, i, j] = self.amplitude * np.sin(2 * np.pi * j * self.frequency / w)
        triggered_image = image + signal.to(image.device)
        return torch.clamp(triggered_image, 0, 1)

# =============================================================================
# 2. Data Handling and Poisoning
# =============================================================================

class PoisonedDataset(Dataset):
    """A wrapper dataset to apply backdoor attacks on the fly."""
    def __init__(self, dataset, attack, poison_rate, poison_target_only=True):
        self.dataset = dataset
        self.attack = attack
        self.poison_rate = poison_rate
        self.poison_target_only = poison_target_only
        
        try:
            self.num_classes = len(dataset.classes)
        except AttributeError:
            self.num_classes = len(np.unique([target for _, target in dataset]))


        self.data, self.targets = [], []
        # Handle Subset case
        if isinstance(dataset, torch.utils.data.Subset):
            for idx in dataset.indices:
                d, t = dataset.dataset[idx]
                self.data.append(d)
                self.targets.append(t)
        else:
            for d, t in self.dataset:
                self.data.append(d)
                self.targets.append(t)
        self.targets = torch.tensor(self.targets)

        if self.poison_target_only:
            candidate_indices = torch.where(self.targets != self.attack.target_class)[0]
        else:
             candidate_indices = torch.arange(len(self.dataset))

        num_to_poison = int(len(candidate_indices) * self.poison_rate)
        self.poison_indices = np.random.choice(candidate_indices.numpy(), num_to_poison, replace=False)
        self.poison_mask = torch.zeros(len(self.dataset), dtype=torch.bool)
        self.poison_mask[self.poison_indices] = True
        print(f"Poisoning {num_to_poison} of {len(candidate_indices)} candidate samples ({self.poison_rate * 100:.2f}%). Target: {self.attack.target_class}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        is_poisoned = self.poison_mask[index].item()

        if is_poisoned:
            image_copy = image.clone()
            image = self.attack.apply_trigger(image_copy)
            label = self.attack.target_class
        
        return image, label, is_poisoned

def get_datasets(dataset_name='cifar10', data_dir='./data'):
    """Fetches and transforms the specified dataset."""
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")
    
    return train_dataset, test_dataset, num_classes
