#!/usr/bin/env python3
"""
REA-Cycle++: Relation-Equivariant, Invertible Adapters with Group/Composition Constraints
Main experimental script that orchestrates training and evaluation.
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import numpy as np

from train import TrainConfig, train_rea_cycle_plus
from evaluate import comprehensive_evaluation

def main():
    """Main experimental pipeline for REA-Cycle++."""
    print("="*60)
    print("REA-CYCLE++: CURING THE REVERSAL CURSE")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    save_dir = ".research/iteration1/images"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")
    
    config = TrainConfig(
        d_model=64,
        k=16,
        batch_size=32,
        epochs=3,
        lr=5e-3,
        weight_decay=0.01,
        alpha_min=0.35,
        alpha_start=1.0,
        K=50,
        use_residual=False,
        relation_names=("->R1",)
    )
    
    print("Configuration:")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Entity subspace dimension: {config.k}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Use residual: {config.use_residual}")
    print(f"  Relations: {config.relation_names}")
    print()
    
    print("PHASE 1: TRAINING REA-CYCLE++ MODEL")
    print("-" * 40)
    start_time = time.time()
    
    try:
        results = train_rea_cycle_plus(config, save_dir)
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        print("\nPHASE 2: COMPREHENSIVE EVALUATION")
        print("-" * 40)
        eval_start = time.time()
        
        eval_results = comprehensive_evaluation(results, config, save_dir)
        eval_time = time.time() - eval_start
        print(f"\nEvaluation completed in {eval_time:.2f} seconds")
        
        summary = {
            'config': {
                'd_model': config.d_model,
                'k': config.k,
                'batch_size': config.batch_size,
                'epochs': config.epochs,
                'lr': config.lr,
                'use_residual': config.use_residual,
                'relation_names': list(config.relation_names)
            },
            'training_time': training_time,
            'evaluation_time': eval_time,
            'results': {
                'forward_accuracy': eval_results['accuracy']['forward_accuracy'],
                'reverse_accuracy': eval_results['accuracy']['reverse_accuracy'],
                'reversal_gap': eval_results['accuracy']['reversal_gap'],
                'avg_composition_error': eval_results['composition']['avg_composition_error'],
                'avg_cycle_error': eval_results['composition']['avg_cycle_error'],
                'avg_orthogonality_error': float(np.mean(eval_results['orthogonality']['orthogonality_errors'])),
                'avg_spectral_norm': float(np.mean(eval_results['orthogonality']['spectral_norms']))
            }
        }
        
        summary_path = os.path.join(save_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nExperiment summary saved to: {summary_path}")
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Key Results:")
        print(f"  Forward Accuracy: {eval_results['accuracy']['forward_accuracy']:.4f}")
        print(f"  Reverse Accuracy: {eval_results['accuracy']['reverse_accuracy']:.4f}")
        print(f"  Reversal Gap: {eval_results['accuracy']['reversal_gap']:.4f}")
        print(f"  Composition Error: {eval_results['composition']['avg_composition_error']:.6f}")
        print(f"  Cycle Error: {eval_results['composition']['avg_cycle_error']:.6f}")
        print()
        print("Generated Plots:")
        plots = [
            "training_loss.pdf",
            "forward_reverse_accuracy.pdf", 
            "composition_cycle_errors.pdf",
            "orthogonality_metrics.pdf"
        ]
        for plot in plots:
            plot_path = os.path.join(save_dir, plot)
            if os.path.exists(plot_path):
                print(f"  ✓ {plot}")
            else:
                print(f"  ✗ {plot} (missing)")
        
        print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Experiment failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def set_status_stopped():
    """Set status_enum to 'stopped' as required."""
    try:
        status_files = [
            ".research/status.json",
            "status.json",
            ".status.json"
        ]
        
        status_data = {"status_enum": "stopped"}
        
        status_file = None
        for sf in status_files:
            if os.path.exists(sf):
                status_file = sf
                break
        
        if status_file is None:
            os.makedirs(".research", exist_ok=True)
            status_file = ".research/status.json"
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print(f"Status set to 'stopped' in {status_file}")
        
    except Exception as e:
        print(f"Warning: Could not set status to 'stopped': {e}")

if __name__ == "__main__":
    success = main()
    
    set_status_stopped()
    
    if success:
        print("\n🎉 REA-Cycle++ experiment completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ REA-Cycle++ experiment failed!")
        sys.exit(1)
