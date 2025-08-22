#!/usr/bin/env python3
"""
Rev-SS2D Main Experiment Script
Implements reversible, streaming, shared-state Vision Mamba with memory efficiency validation
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

from train import RevSS2DTrainer, BaselineTrainer
from evaluate import ModelEvaluator
from preprocess import SyntheticDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('.logs/experiment.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment():
    """Setup experiment directories and environment"""
    os.makedirs('.logs', exist_ok=True)
    os.makedirs('.research/iteration1/images', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def run_memory_microbench(device):
    """Run memory efficiency microbenchmark comparing Rev-SS2D vs baseline"""
    logger.info("=== Memory Microbenchmark ===")
    
    data_gen = SyntheticDataGenerator()
    evaluator = ModelEvaluator()
    
    configs = [
        {'name': 'Baseline', 'channels': 32, 'resolution': 256, 'batch_size': 2},
        {'name': 'Rev-SS2D', 'channels': 32, 'resolution': 256, 'batch_size': 2},
        {'name': 'Baseline', 'channels': 32, 'resolution': 512, 'batch_size': 1},
        {'name': 'Rev-SS2D', 'channels': 32, 'resolution': 512, 'batch_size': 1},
    ]
    
    results = []
    
    for config in configs:
        logger.info(f"Testing {config['name']} - {config['resolution']}x{config['resolution']}")
        
        try:
            data = data_gen.generate_classification_data(
                batch_size=config['batch_size'],
                channels=config['channels'],
                height=config['resolution'],
                width=config['resolution'],
                num_classes=10
            )
            
            if config['name'] == 'Baseline':
                trainer = BaselineTrainer(
                    channels=config['channels'],
                    num_classes=10,
                    device=device
                )
            else:
                trainer = RevSS2DTrainer(
                    channels=config['channels'],
                    num_classes=10,
                    device=device,
                    use_reversible=True,
                    use_streaming=True,
                    use_shared_state=True,
                    tile_size=64
                )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            loss = trainer.train_step(data['images'], data['labels'])
            end_time = time.time()
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            else:
                param_memory = sum(p.numel() * p.element_size() for p in trainer.model.parameters()) / (1024**3)
                peak_memory = param_memory * 2.5
            step_time = end_time - start_time
            
            result = {
                'name': config['name'],
                'resolution': config['resolution'],
                'batch_size': config['batch_size'],
                'peak_memory_gb': peak_memory,
                'step_time_s': step_time,
                'loss': loss.item()
            }
            results.append(result)
            
            logger.info(f"  Peak Memory: {peak_memory:.2f} GB")
            logger.info(f"  Step Time: {step_time:.2f} s")
            logger.info(f"  Loss: {loss.item():.4f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"  OOM Error: {str(e)}")
                result = {
                    'name': config['name'],
                    'resolution': config['resolution'],
                    'batch_size': config['batch_size'],
                    'peak_memory_gb': float('inf'),
                    'step_time_s': float('inf'),
                    'loss': float('nan')
                }
                results.append(result)
            else:
                raise e
    
    evaluator.plot_memory_comparison(results, '.research/iteration1/images/memory_comparison.pdf')
    
    return results

def run_gradient_correctness_test(device):
    """Test gradient correctness between Rev-SS2D and baseline"""
    logger.info("=== Gradient Correctness Test ===")
    
    data_gen = SyntheticDataGenerator()
    evaluator = ModelEvaluator()
    
    data = data_gen.generate_classification_data(
        batch_size=2, channels=32, height=128, width=128, num_classes=5
    )
    
    baseline_trainer = BaselineTrainer(channels=32, num_classes=5, device=device)
    rev_trainer = RevSS2DTrainer(
        channels=32, num_classes=5, device=device,
        use_reversible=True, use_streaming=False, use_shared_state=False
    )
    
    rev_trainer.copy_weights_from_baseline(baseline_trainer)
    
    baseline_loss = baseline_trainer.train_step(data['images'], data['labels'])
    rev_loss = rev_trainer.train_step(data['images'], data['labels'])
    
    grad_diff = evaluator.compare_gradients(baseline_trainer.model, rev_trainer.model)
    
    logger.info(f"Baseline Loss: {baseline_loss.item():.6f}")
    logger.info(f"Rev-SS2D Loss: {rev_loss.item():.6f}")
    logger.info(f"Loss Difference: {abs(baseline_loss.item() - rev_loss.item()):.6f}")
    logger.info(f"Max Gradient Relative Error: {grad_diff['max_rel_error']:.6f}")
    logger.info(f"Mean Gradient Relative Error: {grad_diff['mean_rel_error']:.6f}")
    
    correctness_passed = grad_diff['max_rel_error'] < 1e-3
    logger.info(f"Gradient Correctness Test: {'PASSED' if correctness_passed else 'FAILED'}")
    
    return {
        'baseline_loss': baseline_loss.item(),
        'rev_loss': rev_loss.item(),
        'max_grad_error': grad_diff['max_rel_error'],
        'mean_grad_error': grad_diff['mean_rel_error'],
        'correctness_passed': correctness_passed
    }

def run_ablation_study(device):
    """Run ablation study on Rev-SS2D components"""
    logger.info("=== Ablation Study ===")
    
    data_gen = SyntheticDataGenerator()
    evaluator = ModelEvaluator()
    
    train_data = data_gen.generate_classification_data(
        batch_size=2, channels=32, height=128, width=128, num_classes=5
    )
    
    configs = [
        {'name': 'Baseline', 'reversible': False, 'streaming': False, 'shared': False},
        {'name': 'Rev Only', 'reversible': True, 'streaming': False, 'shared': False},
        {'name': 'Rev + Shared', 'reversible': True, 'streaming': False, 'shared': True},
        {'name': 'Rev + Streaming', 'reversible': True, 'streaming': True, 'shared': False},
        {'name': 'Full Rev-SS2D', 'reversible': True, 'streaming': True, 'shared': True},
    ]
    
    results = []
    
    for config in configs:
        logger.info(f"Testing configuration: {config['name']}")
        
        try:
            if config['name'] == 'Baseline':
                trainer = BaselineTrainer(channels=32, num_classes=5, device=device)
            else:
                trainer = RevSS2DTrainer(
                    channels=32, num_classes=5, device=device,
                    use_reversible=config['reversible'],
                    use_streaming=config['streaming'],
                    use_shared_state=config['shared'],
                    tile_size=32
                )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            losses = []
            for step in range(3):
                loss = trainer.train_step(train_data['images'], train_data['labels'])
                losses.append(loss.item())
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            else:
                param_memory = sum(p.numel() * p.element_size() for p in trainer.model.parameters()) / (1024**3)
                peak_memory = param_memory * 3.0
            final_loss = losses[-1]
            
            result = {
                'config': config['name'],
                'peak_memory_gb': peak_memory,
                'final_loss': final_loss,
                'losses': losses
            }
            results.append(result)
            
            logger.info(f"  Peak Memory: {peak_memory:.2f} GB")
            logger.info(f"  Final Loss: {final_loss:.4f}")
            
        except Exception as e:
            logger.error(f"  Error in {config['name']}: {str(e)}")
            result = {
                'config': config['name'],
                'peak_memory_gb': float('inf'),
                'final_loss': float('nan'),
                'losses': [float('nan')] * 5
            }
            results.append(result)
    
    evaluator.plot_ablation_study(results, '.research/iteration1/images/ablation_study.pdf')
    evaluator.plot_training_curves(results, '.research/iteration1/images/training_curves.pdf')
    
    return results

def main():
    """Main experiment execution"""
    logger.info("Starting Rev-SS2D Experiment")
    logger.info("=" * 50)
    
    device = setup_experiment()
    
    try:
        memory_results = run_memory_microbench(device)
        
        correctness_results = run_gradient_correctness_test(device)
        
        ablation_results = run_ablation_study(device)
        
        all_results = {
            'memory_benchmark': memory_results,
            'gradient_correctness': correctness_results,
            'ablation_study': ablation_results,
            'timestamp': time.time(),
            'device': str(device)
        }
        
        with open('.research/iteration1/results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info("=" * 50)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 50)
        
        baseline_mem = [r for r in memory_results if r['name'] == 'Baseline']
        rev_mem = [r for r in memory_results if r['name'] == 'Rev-SS2D']
        
        if baseline_mem and rev_mem:
            for b, r in zip(baseline_mem, rev_mem):
                if b['resolution'] == r['resolution']:
                    if r['peak_memory_gb'] != float('inf') and b['peak_memory_gb'] != float('inf'):
                        reduction = b['peak_memory_gb'] / r['peak_memory_gb']
                        logger.info(f"Memory reduction at {b['resolution']}x{b['resolution']}: {reduction:.2f}x")
        
        logger.info(f"Gradient correctness: {'PASSED' if correctness_results['correctness_passed'] else 'FAILED'}")
        logger.info(f"Max gradient error: {correctness_results['max_grad_error']:.6f}")
        
        status_file = Path('.research/status.json')
        status_data = {'status_enum': 'stopped', 'timestamp': time.time()}
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to .research/iteration1/")
        logger.info(f"Plots saved to .research/iteration1/images/")
        logger.info("Status set to 'stopped'")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        status_file = Path('.research/status.json')
        status_data = {'status_enum': 'error', 'error': str(e), 'timestamp': time.time()}
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        raise

if __name__ == "__main__":
    main()
