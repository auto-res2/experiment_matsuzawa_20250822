import os
import json
import time
import torch
import argparse
from typing import Dict, Any

from preprocess import prepare_discrete_experiment, prepare_continuous_experiment, get_data_statistics
from train import TinySSMModel, train_model, set_seed
from evaluate import comprehensive_evaluation


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def run_discrete_experiment(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    print("\n" + "="*60)
    print("RUNNING DISCRETE SEQUENCE EXPERIMENT")
    print("="*60)
    
    train_loader, val_loader, test_loader, vocab_size = prepare_discrete_experiment(
        num_samples=config['num_samples'],
        seq_len=config['seq_len'],
        vocab_size=config['vocab_size'],
        pattern_complexity=config['pattern_complexity'],
        tiny_fraction=config['tiny_fraction'],
        batch_size=config['batch_size'],
        seed=config['seed']
    )
    
    print("\nDataset Statistics:")
    train_stats = get_data_statistics(train_loader, 'discrete')
    print(f"Training samples: {train_stats['total_samples']}")
    print(f"Label distribution: {train_stats['label_distribution']}")
    print(f"Vocabulary size: {train_stats['vocab_size']}")
    
    model = TinySSMModel(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_modes=config['n_modes'],
        kernel_len=config['kernel_len'],
        num_classes=2
    )
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nStarting training with ID-SPT++ method...")
    start_time = time.time()
    
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate']
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    print("\nRunning comprehensive evaluation...")
    eval_results = comprehensive_evaluation(
        model=results['model'],
        test_loader=test_loader,
        device=device,
        train_losses=results['train_losses'],
        val_losses=results['val_losses'],
        val_accuracies=results['val_accuracies'],
        output_dir='.research/iteration1/images'
    )
    
    return {
        'experiment_type': 'discrete',
        'config': config,
        'training_results': results,
        'evaluation_results': eval_results,
        'training_time': training_time,
        'final_accuracy': eval_results['evaluation']['accuracy']
    }


def run_continuous_experiment(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    print("\n" + "="*60)
    print("RUNNING CONTINUOUS SIGNAL EXPERIMENT")
    print("="*60)
    
    train_loader, val_loader, test_loader = prepare_continuous_experiment(
        num_samples=config['num_samples'],
        seq_len=config['seq_len'],
        signal_type=config['signal_type'],
        tiny_fraction=config['tiny_fraction'],
        batch_size=config['batch_size'],
        seed=config['seed']
    )
    
    print("\nDataset Statistics:")
    train_stats = get_data_statistics(train_loader, 'continuous')
    print(f"Training samples: {train_stats['total_samples']}")
    print(f"Label distribution: {train_stats['label_distribution']}")
    print(f"Signal statistics: mean={train_stats['signal_mean']:.3f}, std={train_stats['signal_std']:.3f}")
    
    model = TinySSMModel(
        vocab_size=1,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_modes=config['n_modes'],
        kernel_len=config['kernel_len'],
        num_classes=2
    )
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nStarting training with ID-SPT++ method...")
    start_time = time.time()
    
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate']
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    print("\nRunning comprehensive evaluation...")
    eval_results = comprehensive_evaluation(
        model=results['model'],
        test_loader=test_loader,
        device=device,
        train_losses=results['train_losses'],
        val_losses=results['val_losses'],
        val_accuracies=results['val_accuracies'],
        output_dir='.research/iteration1/images'
    )
    
    return {
        'experiment_type': 'continuous',
        'config': config,
        'training_results': results,
        'evaluation_results': eval_results,
        'training_time': training_time,
        'final_accuracy': eval_results['evaluation']['accuracy']
    }


def run_quick_test(device: torch.device) -> bool:
    print("\n" + "="*60)
    print("RUNNING QUICK FUNCTIONALITY TEST")
    print("="*60)
    
    try:
        print("Testing discrete sequence processing...")
        train_loader, val_loader, test_loader, vocab_size = prepare_discrete_experiment(
            num_samples=100,
            seq_len=64,
            vocab_size=20,
            pattern_complexity='simple',
            tiny_fraction=0.1,
            batch_size=8,
            seed=42
        )
        
        model = TinySSMModel(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=2,
            n_modes=4,
            kernel_len=32,
            num_classes=2
        )
        
        print("Testing forward pass...")
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch['tokens'])
                print(f"Input shape: {batch['tokens'].shape}, Output shape: {logits.shape}")
                break
        
        print("Testing training step...")
        try:
            results = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=1,
                lr=1e-3
            )
            print("Training completed successfully!")
        except Exception as e:
            import traceback
            print(f"Training failed: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            raise e
        
        print("Testing evaluation...")
        try:
            eval_results = comprehensive_evaluation(
                model=results['model'],
                test_loader=test_loader,
                device=device,
                train_losses=results['train_losses'],
                val_losses=results['val_losses'],
                val_accuracies=results['val_accuracies'],
                output_dir='.research/iteration1/images'
            )
            print("Evaluation completed successfully!")
        except Exception as e:
            print(f"Evaluation failed: {e}")
            raise e
        
        print("✓ Quick test passed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='ID-SPT++ Experiments')
    parser.add_argument('--mode', choices=['test', 'discrete', 'continuous', 'all'], 
                       default='test', help='Experiment mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("ID-SPT++: Identifiable, Data-Driven Self-Pretraining for SSMs")
    print("=" * 60)
    
    set_seed(args.seed)
    device = setup_device()
    
    base_config = {
        'seed': args.seed,
        'num_epochs': args.epochs,
        'learning_rate': 1e-3,
        'batch_size': 16,
        'd_model': 128,
        'n_layers': 4,
        'n_modes': 8,
        'kernel_len': 64,
        'tiny_fraction': 0.05
    }
    
    results = {}
    
    if args.mode in ['test', 'all']:
        test_passed = run_quick_test(device)
        results['test_passed'] = test_passed
        
        if not test_passed and args.mode == 'test':
            print("\nTest failed. Exiting.")
            return
    
    if args.mode in ['discrete', 'all']:
        discrete_config = base_config.copy()
        discrete_config.update({
            'num_samples': 1000,
            'seq_len': 128,
            'vocab_size': 50,
            'pattern_complexity': 'medium'
        })
        
        discrete_results = run_discrete_experiment(discrete_config, device)
        results['discrete'] = discrete_results
    
    if args.mode in ['continuous', 'all']:
        continuous_config = base_config.copy()
        continuous_config.update({
            'num_samples': 1000,
            'seq_len': 256,
            'signal_type': 'mixed'
        })
        
        continuous_results = run_continuous_experiment(continuous_config, device)
        results['continuous'] = continuous_results
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    if 'discrete' in results:
        print(f"Discrete Experiment - Final Accuracy: {results['discrete']['final_accuracy']:.4f}")
    
    if 'continuous' in results:
        print(f"Continuous Experiment - Final Accuracy: {results['continuous']['final_accuracy']:.4f}")
    
    results_file = '.research/iteration1/experiment_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'results': results,
            'status_enum': 'stopped'
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    print("Experiment completed successfully!")


if __name__ == '__main__':
    main()
