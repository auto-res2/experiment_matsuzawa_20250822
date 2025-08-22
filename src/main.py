#!/usr/bin/env python3
"""
DASH-HiLo-Anchor for SHViT — Main experimental script
Implements the complete experimental workflow from preprocessing to evaluation.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import create_datasets, SyntheticObjectsDataset
from train import train_model, DASHHiLoSHViT
from evaluate import run_experiments

def setup_environment():
    """Setup experimental environment and paths."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    os.makedirs('.research/iteration1/images', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    return device

def main():
    """Main experimental workflow."""
    print("=" * 80)
    print("DASH-HiLo-Anchor for SHViT Experimental Framework")
    print("=" * 80)
    
    device = setup_environment()
    
    config = {
        'image_size': 96,
        'num_classes': 4,
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 1e-3,
        'device': device,
        'save_dir': 'models',
        'results_dir': '.research/iteration1/images'
    }
    
    print(f"Configuration: {config}")
    print()
    
    print("Step 1: Data Preprocessing")
    print("-" * 40)
    train_loader, val_loader, test_loader = create_datasets(
        image_size=config['image_size'],
        batch_size=config['batch_size']
    )
    print(f"Created datasets - Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    print()
    
    print("Step 2: Model Training")
    print("-" * 40)
    
    models = {}
    
    print("Training Baseline SHViT...")
    baseline_model = DASHHiLoSHViT(
        num_classes=config['num_classes'],
        enable_dash=False,
        enable_hilo=False,
        enable_anchor=False
    )
    baseline_model, baseline_history = train_model(
        baseline_model, train_loader, val_loader, config, model_name="baseline"
    )
    models['baseline'] = (baseline_model, baseline_history)
    
    print("Training DASH variant...")
    dash_model = DASHHiLoSHViT(
        num_classes=config['num_classes'],
        enable_dash=True,
        enable_hilo=False,
        enable_anchor=False
    )
    dash_model, dash_history = train_model(
        dash_model, train_loader, val_loader, config, model_name="dash"
    )
    models['dash'] = (dash_model, dash_history)
    
    print("Training HiLo variant...")
    hilo_model = DASHHiLoSHViT(
        num_classes=config['num_classes'],
        enable_dash=False,
        enable_hilo=True,
        enable_anchor=False
    )
    hilo_model, hilo_history = train_model(
        hilo_model, train_loader, val_loader, config, model_name="hilo"
    )
    models['hilo'] = (hilo_model, hilo_history)
    
    print("Training Full DASH-HiLo-Anchor...")
    full_model = DASHHiLoSHViT(
        num_classes=config['num_classes'],
        enable_dash=True,
        enable_hilo=True,
        enable_anchor=True
    )
    full_model, full_history = train_model(
        full_model, train_loader, val_loader, config, model_name="full"
    )
    models['full'] = (full_model, full_history)
    
    print()
    
    print("Step 3: Evaluation and Experiments")
    print("-" * 40)
    
    results = run_experiments(models, test_loader, config)
    
    results_file = os.path.join(config['results_dir'], 'experimental_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    print()
    print("Step 4: Setting Status")
    print("-" * 40)
    
    status_data = {
        'status_enum': 'stopped',
        'completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiments_completed': True,
        'models_trained': list(models.keys()),
        'results_saved': True
    }
    
    status_file = os.path.join(config['results_dir'], 'experiment_status.json')
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    print(f"Status set to 'stopped' and saved to: {status_file}")
    
    print()
    print("=" * 80)
    print("DASH-HiLo-Anchor Experimental Framework Complete!")
    print("=" * 80)
    print(f"All results and plots saved to: {config['results_dir']}")
    print("Key findings:")
    for variant, result in results.items():
        if 'accuracy' in result:
            print(f"  {variant}: {result['accuracy']:.3f} accuracy")
    print()

if __name__ == "__main__":
    main()
