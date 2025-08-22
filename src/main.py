"""
Main script for SEEDS experiments
"""
import os
import sys
import torch
import json
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.seeds_config import config
from src.preprocess import main as preprocess_main
from src.train import main as train_main
from src.evaluate import main as evaluate_main

def set_status(status: str):
    """Set experiment status"""
    status_file = os.path.join(config.results_dir, 'status.json')
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    
    status_data = {
        'status_enum': status,
        'timestamp': datetime.now().isoformat(),
        'experiment': 'SEEDS'
    }
    
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    print(f"Status set to: {status}")

def print_experiment_info():
    """Print experiment information"""
    print("=" * 80)
    print("SEEDS — Surrogate‑Emulated, Event‑Driven Sampling for Fast Discrete Diffusion")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")
    print(f"K (states): {config.K}")
    print(f"Image size: {config.image_size}x{config.image_size}")
    print(f"Sequence length: {config.seq_length}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.epochs}")
    print(f"Delta (violation rate): {config.delta}")
    print(f"Results directory: {config.results_dir}")
    print("=" * 80)

def main():
    """Main experiment pipeline"""
    start_time = time.time()
    
    try:
        set_status("running")
        
        print_experiment_info()
        
        if config.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            config.device = "cpu"
        
        print(f"Using device: {config.device}")
        
        print("\n" + "="*50)
        print("STEP 1: DATA PREPROCESSING")
        print("="*50)
        
        datasets = preprocess_main()
        print("✓ Data preprocessing completed")
        
        print("\n" + "="*50)
        print("STEP 2: MODEL TRAINING")
        print("="*50)
        
        training_results = train_main()
        print("✓ Model training completed")
        
        print("\n" + "="*50)
        print("STEP 3: EVALUATION")
        print("="*50)
        
        evaluation_results, summary_stats = evaluate_main()
        print("✓ Evaluation completed")
        
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        
        total_time = time.time() - start_time
        print(f"Total runtime: {total_time:.2f} seconds")
        
        if summary_stats:
            print("\nKey Results:")
            for data_type, methods_data in summary_stats.items():
                print(f"\n{data_type.title()} Data:")
                for method, stats in methods_data.items():
                    method_name = {
                        'seeds_exact': 'SEEDS (Exact)',
                        'seeds_budgeted': 'SEEDS (Budgeted)', 
                        'tau_leaping': 'Tau-leaping'
                    }.get(method, method)
                    
                    nfe = stats['nfe_heavy_mean']
                    time_val = stats['wall_time_mean']
                    events = stats['n_events_mean']
                    
                    print(f"  {method_name:20}: NFE={nfe:5.1f}, Time={time_val:6.3f}s, Events={events:5.1f}")
        
        print("\nSpeedup Analysis:")
        for data_type, methods_data in summary_stats.items():
            if 'tau_leaping' in methods_data:
                baseline_nfe = methods_data['tau_leaping']['nfe_heavy_mean']
                baseline_time = methods_data['tau_leaping']['wall_time_mean']
                
                print(f"\n{data_type.title()} vs Tau-leaping:")
                
                for method in ['seeds_exact', 'seeds_budgeted']:
                    if method in methods_data:
                        method_nfe = methods_data[method]['nfe_heavy_mean']
                        method_time = methods_data[method]['wall_time_mean']
                        
                        nfe_speedup = baseline_nfe / method_nfe if method_nfe > 0 else float('inf')
                        time_speedup = baseline_time / method_time if method_time > 0 else float('inf')
                        
                        method_name = {
                            'seeds_exact': 'SEEDS (Exact)',
                            'seeds_budgeted': 'SEEDS (Budgeted)'
                        }[method]
                        
                        print(f"  {method_name:20}: {nfe_speedup:4.1f}x NFE, {time_speedup:4.1f}x Time")
        
        print(f"\nResults saved to: {config.results_dir}")
        print("Generated files:")
        
        if os.path.exists(config.results_dir):
            for file in os.listdir(config.results_dir):
                if file.endswith('.pdf'):
                    print(f"  - {file}")
        
        set_status("stopped")
        
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'summary_stats': summary_stats,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"\nERROR: Experiment failed with exception: {e}")
        import traceback
        traceback.print_exc()
        
        set_status("error")
        
        raise e

if __name__ == "__main__":
    results = main()
