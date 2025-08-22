"""
FOST-PEFT Main Experiment Script
Orchestrates the complete FOST-PEFT experiment from data preprocessing to evaluation.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any

import torch
import numpy as np

from preprocess import generate_synthetic_stream, create_dataloaders, prepare_cifar_stream, prepare_text_stream
from train import FOSTModel, train_fost_model
from evaluate import evaluate_continual_learning, plot_training_curves, plot_evaluation_results, create_confusion_matrix


def ensure_dir(path: str):
    """Ensure directory exists."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def set_status(status: str, config_dir: str = "./config"):
    """Set experiment status."""
    ensure_dir(config_dir)
    status_file = os.path.join(config_dir, "status.json")
    
    status_data = {"status_enum": status, "timestamp": time.time()}
    
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    print(f"Status set to: {status}")


def run_synthetic_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run synthetic data experiment for quick testing.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with experiment results
    """
    print("=" * 60)
    print("FOST-PEFT SYNTHETIC EXPERIMENT")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\nGenerating synthetic continual learning stream...")
    print(f"  Tasks: {args.n_tasks}")
    print(f"  Samples per task: {args.samples_per_task}")
    print(f"  Input dimension: {args.input_dim}")
    print(f"  Classes: {args.n_classes}")
    print(f"  LoRA rank: {args.lora_rank}")
    
    tasks = generate_synthetic_stream(
        n_tasks=args.n_tasks,
        samples_per_task=args.samples_per_task,
        input_dim=args.input_dim,
        n_classes=args.n_classes,
        drift_strength=0.3,
        seed=args.seed
    )
    
    dataloaders = create_dataloaders(tasks, batch_size=args.batch_size, shuffle=True)
    
    print(f"\nCreating FOST-PEFT model...")
    model = FOSTModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.n_classes,
        r=args.lora_rank
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter efficiency: {100.0 * trainable_params / total_params:.2f}%")
    
    print(f"\nTraining FOST-PEFT model...")
    start_time = time.time()
    
    training_metrics = train_fost_model(
        model=model,
        dataloaders=dataloaders,
        n_epochs=args.epochs_per_task,
        lr=args.learning_rate,
        device=device
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    print(f"\nEvaluating FOST-PEFT model...")
    eval_metrics = evaluate_continual_learning(model, dataloaders, device)
    
    print(f"\nGenerating plots...")
    images_dir = ".research/iteration1/images"
    ensure_dir(images_dir)
    
    plot_training_curves(training_metrics, save_dir=images_dir)
    plot_evaluation_results(eval_metrics, save_dir=images_dir)
    create_confusion_matrix(model, dataloaders[-1], args.n_classes, 
                          save_dir=images_dir, task_name="final", device=device)
    
    results = {
        'experiment_type': 'synthetic',
        'config': vars(args),
        'training_metrics': training_metrics,
        'evaluation_metrics': eval_metrics,
        'training_time': training_time,
        'device': device,
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'parameter_efficiency': 100.0 * trainable_params / total_params
        }
    }
    
    return results


def run_cifar_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run CIFAR-100 class-incremental experiment (placeholder).
    For quick test, this uses synthetic data.
    """
    print("=" * 60)
    print("FOST-PEFT CIFAR-100 EXPERIMENT (SYNTHETIC FOR QUICK TEST)")
    print("=" * 60)
    
    args.input_dim = 512  # Simulated CNN feature dimension
    args.n_classes = 20   # 20 classes per task for 5 tasks = 100 classes
    args.samples_per_task = 200
    
    return run_synthetic_experiment(args)


def run_text_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run text domain adaptation experiment (placeholder).
    For quick test, this uses synthetic data.
    """
    print("=" * 60)
    print("FOST-PEFT TEXT DOMAIN EXPERIMENT (SYNTHETIC FOR QUICK TEST)")
    print("=" * 60)
    
    args.input_dim = 768   # Simulated BERT embedding dimension
    args.n_classes = 5     # Sentiment classes
    args.samples_per_task = 150
    args.n_tasks = 3       # 3 domains
    
    return run_synthetic_experiment(args)


def main():
    """Main experiment orchestration."""
    parser = argparse.ArgumentParser(description='FOST-PEFT Continual Learning Experiment')
    
    parser.add_argument('--experiment', type=str, default='synthetic',
                       choices=['synthetic', 'cifar', 'text'],
                       help='Type of experiment to run')
    
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank (r)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden layer dimension')
    
    parser.add_argument('--n_tasks', type=int, default=5,
                       help='Number of tasks in continual learning stream')
    parser.add_argument('--samples_per_task', type=int, default=500,
                       help='Number of samples per task')
    parser.add_argument('--input_dim', type=int, default=128,
                       help='Input feature dimension')
    parser.add_argument('--n_classes', type=int, default=10,
                       help='Number of classes')
    
    parser.add_argument('--epochs_per_task', type=int, default=5,
                       help='Training epochs per task')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage (ignore CUDA)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with minimal parameters')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("QUICK TEST MODE ENABLED - Using minimal parameters")
        args.n_tasks = 3
        args.samples_per_task = 100
        args.epochs_per_task = 2
        args.batch_size = 16
        args.input_dim = 64
        args.hidden_dim = 32
        args.lora_rank = 4
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    set_status("running")
    
    try:
        if args.experiment == 'synthetic':
            results = run_synthetic_experiment(args)
        elif args.experiment == 'cifar':
            results = run_cifar_experiment(args)
        elif args.experiment == 'text':
            results = run_text_experiment(args)
        else:
            raise ValueError(f"Unknown experiment type: {args.experiment}")
        
        results_file = "./config/experiment_results.json"
        ensure_dir("./config")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {results_file}")
        print(f"Plots saved to: .research/iteration1/images/")
        
        eval_metrics = results['evaluation_metrics']
        print(f"\nKey Results:")
        print(f"  Average Accuracy: {eval_metrics['average_accuracy']:.2f}%")
        print(f"  Final Task Accuracy: {eval_metrics['final_accuracy']:.2f}%")
        print(f"  Backward Transfer: {eval_metrics['backward_transfer']:.2f}")
        print(f"  Forward Transfer: {eval_metrics['forward_transfer']:.2f}")
        print(f"  Forgetting: {eval_metrics['forgetting']:.2f}")
        print(f"  Training Time: {results['training_time']:.2f}s")
        print(f"  Parameter Efficiency: {results['model_info']['parameter_efficiency']:.2f}%")
        
        set_status("stopped")
        
    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        set_status("error")
        sys.exit(1)


if __name__ == "__main__":
    main()
