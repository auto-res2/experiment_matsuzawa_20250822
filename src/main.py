"""
CCAD-KD Main Experiment Script
Orchestrates the complete CCAD-KD experiment pipeline from data preparation to evaluation.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any

import torch
import numpy as np

from .preprocess import seed_all, get_device, ensure_dir, get_cifar100_loaders
from .train import train_ccad_kd, create_models
from .evaluate import evaluate_ccad_model, CCADEvaluator


def update_status(status: str, config_path: str = './config/experiment_config.json'):
    """Update experiment status."""
    ensure_dir(os.path.dirname(config_path))
    
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except:
            pass
    
    config['status_enum'] = status
    config['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Status updated to: {status}")


def run_quick_test():
    """Run a quick test to verify the implementation works."""
    print("Running quick test...")
    
    device = get_device()
    seed_all(42)
    
    batch_size = 16
    num_samples = 64
    img_size = 32  # Smaller for faster testing
    num_classes = 10  # Fewer classes for testing
    
    from torch.utils.data import TensorDataset, DataLoader
    from .preprocess import AugMeta
    
    imgs = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    aug_metas = [AugMeta(1.0, 1.0, 1.0, 1.0, 0.0, 0.0) for _ in range(num_samples)]
    
    class TestDataset:
        def __init__(self, imgs, labels, aug_metas):
            self.imgs = imgs
            self.labels = labels
            self.aug_metas = aug_metas
        
        def __len__(self):
            return len(self.imgs)
        
        def __getitem__(self, idx):
            return self.imgs[idx], self.labels[idx], self.aug_metas[idx]
    
    from .preprocess import custom_collate_fn
    
    dataset = TestDataset(imgs, labels, aug_metas)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    import torch.nn as nn
    
    class SimpleNet(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Linear(16, num_classes)
        
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    teacher = SimpleNet(num_classes).to(device)
    student = SimpleNet(num_classes).to(device)
    
    from .train import CCADTrainer
    
    trainer = CCADTrainer(teacher, student, device=device, warmup_epochs=1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    
    print("Testing training loop...")
    for epoch in range(2):
        train_metrics = trainer.train_epoch(loader, optimizer, epoch, 2)
        val_metrics = trainer.validate(loader)
        print(f"Epoch {epoch+1}: Train Loss: {train_metrics['total_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.2f}%")
    
    print("Testing evaluation...")
    evaluator = CCADEvaluator(device=device)
    results = evaluator.evaluate_model(student, loader, fit_context_manager=True)
    print(f"Test Accuracy: {results['accuracy']:.2f}%, ECE: {results['ece']:.4f}")
    
    print("Quick test completed successfully!")
    return True


def run_full_experiment(config: Dict[str, Any]):
    """Run the full CCAD-KD experiment."""
    print("Starting CCAD-KD experiment...")
    print(f"Configuration: {config}")
    
    device = get_device()
    seed_all(config.get('seed', 42))
    
    ensure_dir(config['save_dir'])
    ensure_dir(config['image_dir'])
    
    print("Loading CIFAR-100 dataset...")
    train_loader, val_loader = get_cifar100_loaders(
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config.get('num_workers', 4)
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    print("Starting CCAD-KD training...")
    history = train_ccad_kd(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate'],
        device=device,
        save_dir=config['save_dir']
    )
    
    print("Evaluating trained model...")
    best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
    
    if os.path.exists(best_model_path):
        results = evaluate_ccad_model(
            checkpoint_path=best_model_path,
            test_loader=val_loader,
            model_architecture='resnet18',
            device=device,
            save_dir=config['image_dir']
        )
        
        evaluator = CCADEvaluator(device=device)
        evaluator.plot_training_curves(
            history, 
            os.path.join(config['image_dir'], 'training_curves.pdf')
        )
        
        print("Experiment completed successfully!")
        print(f"Final Results:")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  ECE: {results['ece']:.4f}")
        print(f"  Worst-Group Accuracy: {results['context_analysis']['worst_group_accuracy']:.4f}")
        print(f"  Number of Contexts: {results['context_analysis']['num_contexts']}")
        
        return results
    else:
        print(f"Error: Best model not found at {best_model_path}")
        return None


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='CCAD-KD Experiment')
    parser.add_argument('--test', action='store_true', help='Run quick test only')
    parser.add_argument('--config', type=str, default='./config/experiment_config.json',
                       help='Path to experiment configuration file')
    
    args = parser.parse_args()
    
    update_status('running', args.config)
    
    try:
        if args.test:
            success = run_quick_test()
            if success:
                update_status('test_passed', args.config)
            else:
                update_status('test_failed', args.config)
                return 1
        else:
            default_config = {
                'seed': 42,
                'batch_size': 128,
                'img_size': 224,
                'num_epochs': 20,  # Reduced for Tesla T4 constraints
                'learning_rate': 0.1,
                'num_workers': 4,
                'save_dir': './models',
                'image_dir': './.research/iteration1/images'
            }
            
            config = default_config.copy()
            if os.path.exists(args.config):
                try:
                    with open(args.config, 'r') as f:
                        user_config = json.load(f)
                        config.update(user_config)
                except Exception as e:
                    print(f"Warning: Could not load config file {args.config}: {e}")
                    print("Using default configuration.")
            
            ensure_dir(os.path.dirname(args.config))
            with open(args.config, 'w') as f:
                json.dump(config, f, indent=2)
            
            results = run_full_experiment(config)
            
            if results is not None:
                update_status('stopped', args.config)
            else:
                update_status('failed', args.config)
                return 1
    
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        update_status('failed', args.config)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
