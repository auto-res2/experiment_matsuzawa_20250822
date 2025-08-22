#!/usr/bin/env python3

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from preprocess import create_synthetic_datasets
from train import train_models
from evaluate import evaluate_models, generate_figures

def main():
    print("=" * 80)
    print("ELLA-Regs: Elastic, Layerwise, Learned Allocation of Registers for Vision Transformers")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    results_dir = Path(".research/iteration1/images")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: Local/Global Separation with EVT Outlier Control")
    print("=" * 50)
    
    print("\nStep 1: Creating synthetic datasets...")
    datasets = create_synthetic_datasets()
    
    print("\nStep 2: Training models...")
    models, training_logs = train_models(datasets, device)
    
    print("\nStep 3: Evaluating models and generating figures...")
    results = evaluate_models(models, datasets, device)
    
    print("\nStep 4: Generating academic-quality PDF figures...")
    generate_figures(results, training_logs, results_dir)
    
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: Hardware-Budgeted Elastic Capacity")
    print("=" * 50)
    
    print("Running latency-constrained experiments...")
    from train import run_latency_experiments
    latency_results = run_latency_experiments(datasets, device)
    
    print("\n" + "=" * 50)
    print("EXPERIMENT 3: Nullspace-Aware Tethers for CLIP Robustness")
    print("=" * 50)
    
    print("Running nullspace robustness experiments...")
    from train import run_nullspace_experiments
    nullspace_results = run_nullspace_experiments(datasets, device)
    
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    
    print(f"\nExperiment 1 Results:")
    print(f"  ELLA-Regs Accuracy: {results['ella_accuracy']:.3f}")
    print(f"  Local-Only Accuracy: {results['local_accuracy']:.3f}")
    print(f"  Token-Append Accuracy: {results['token_accuracy']:.3f}")
    print(f"  ELLA-Regs Latency: {results['ella_latency']:.2f}ms")
    print(f"  Token-Append Latency: {results['token_latency']:.2f}ms")
    
    print(f"\nExperiment 2 Results:")
    print(f"  Budget-Constrained Accuracy: {latency_results['constrained_accuracy']:.3f}")
    print(f"  Achieved Latency: {latency_results['achieved_latency']:.2f}ms")
    print(f"  Target Latency: {latency_results['target_latency']:.2f}ms")
    
    print(f"\nExperiment 3 Results:")
    print(f"  With Nullspace Tether: {nullspace_results['with_tether_accuracy']:.3f}")
    print(f"  Without Nullspace Tether: {nullspace_results['without_tether_accuracy']:.3f}")
    print(f"  Spectral Alignment Score: {nullspace_results['alignment_score']:.3f}")
    
    generated_files = list(results_dir.glob("*.pdf"))
    print(f"\nGenerated {len(generated_files)} PDF figures:")
    for file in sorted(generated_files):
        print(f"  - {file.name}")
    
    print("\n" + "=" * 80)
    print("ELLA-Regs experiments completed successfully!")
    print("All results saved to .research/iteration1/images/")
    print("=" * 80)
    
    return {
        'experiment1': results,
        'experiment2': latency_results, 
        'experiment3': nullspace_results,
        'status': 'completed'
    }

if __name__ == "__main__":
    results = main()
