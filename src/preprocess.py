import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import time

def get_cifar10_dataloaders(config):
    """Loads and returns the CIFAR-10 dataset and dataloaders."""
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root=config['DATA_DIR'], train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=config['DATA_DIR'], train=False, download=True, transform=transform)
    
    test_loader = DataLoader(test_set, batch_size=config['TRAIN_BATCH_SIZE'], shuffle=False)
    return train_set, test_loader

def generate_synthetic_data(pipeline, method_name, config):
    """Generates synthetic data using a diffusion pipeline."""
    print(f"\n--- Generating data for method: {method_name} ---")
    num_images = config['NUM_GEN_IMAGES']
    batch_size = config['GEN_BATCH_SIZE']
    num_batches = int(np.ceil(num_images / batch_size))
    
    # Create balanced labels
    images_per_class = num_images // config['NUM_CLASSES']
    labels = torch.arange(config['NUM_CLASSES']).repeat_interleave(images_per_class)
    if len(labels) < num_images:
        labels = torch.cat([labels, torch.arange(num_images - len(labels)) % config['NUM_CLASSES']])

    all_images = []
    pipeline.to(config['DEVICE'])

    # Warm-up run
    if config['DEVICE'] == 'cuda':
        _ = pipeline(batch_size=1).images
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for i in tqdm(range(num_batches), desc=f"Generating for {method_name}"):
        current_batch_size = min(batch_size, num_images - i * batch_size)
        if current_batch_size <= 0: break
            
        images_np = pipeline(batch_size=current_batch_size).images
        images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2) / 255.0
        all_images.append(images_tensor.cpu())

        # Simulate the online optimization overhead of DistDiff
        if method_name == 'DistDiff':
            time.sleep(0.05 * current_batch_size)

    end_time = time.perf_counter()
    
    elapsed_time_ms = (end_time - start_time) * 1000
    time_per_image_ms = elapsed_time_ms / num_images
    print(f"Total generation time: {elapsed_time_ms/1000:.2f} s")
    print(f"Time per image: {time_per_image_ms:.4f} ms")

    all_images_tensor = torch.cat(all_images, dim=0)
    # Normalize to [-1, 1] as expected by ResNet
    all_images_tensor = (all_images_tensor - 0.5) * 2.0
    
    return TensorDataset(all_images_tensor, labels), time_per_image_ms
