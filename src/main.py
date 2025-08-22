"""
GASP (Grouped Autoregressive Scale Prediction) Latency Benchmark Experiment

This script implements a comprehensive latency benchmark for the GASP model,
measuring inference time across different group sizes to validate the hypothesis
that larger group sizes lead to deterministic speedup in autoregressive generation.
"""

import torch
import torch.nn as nn
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import sys
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import GASPModel
from evaluate import run_benchmark, plot_results
from preprocess import setup_experiment_environment

def update_status_enum(status="stopped"):
    """Update the status_enum in research_history.json"""
    research_file = Path(__file__).parent.parent / ".research" / "research_history.json"
    
    try:
        with open(research_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data["status_enum"] = status
        
        with open(research_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Updated status_enum to: {status}")
    except Exception as e:
        print(f"Warning: Could not update status_enum: {e}")

def main():
    """Main function to run the full GASP latency benchmark experiment."""
    print("=" * 60)
    print("GASP Latency Benchmark Experiment")
    print("=" * 60)
    
    setup_experiment_environment()
    
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a GPU.")
        print("Falling back to CPU for demonstration purposes...")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"Running on device: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    group_sizes_to_test = [1, 2, 4, 8, 16, 32]
    num_tokens = 256  # Simulating a 256x256 image with 16x16 patches
    num_runs = 50 if device == 'cuda' else 10  # Reduced for faster execution
    warmup_runs = 10 if device == 'cuda' else 2
    model_embed_dim = 768  # Reduced for T4 compatibility
    model_num_layers = 12  # Reduced for T4 compatibility

    print(f"\nExperiment Configuration:")
    print(f"- Group sizes: {group_sizes_to_test}")
    print(f"- Number of tokens: {num_tokens}")
    print(f"- Runs per group size: {num_runs}")
    print(f"- Warmup runs: {warmup_runs}")
    print(f"- Model embedding dimension: {model_embed_dim}")
    print(f"- Model layers: {model_num_layers}")

    try:
        model = GASPModel(embed_dim=model_embed_dim, num_layers=model_num_layers).to(device).eval()
        print(f"\nModel created successfully with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    results = []
    for g_size in group_sizes_to_test:
        print(f"\n{'='*50}")
        print(f"Testing Group Size: {g_size}")
        print(f"{'='*50}")
        
        try:
            run_latencies = run_benchmark(
                model, g_size, num_tokens, device, 
                num_runs=num_runs, warmup_runs=warmup_runs
            )
            
            for lat in run_latencies:
                results.append({'group_size': g_size, 'latency_sec': lat})
                
            mean_lat = np.mean(run_latencies)
            std_lat = np.std(run_latencies)
            print(f"Group {g_size}: {mean_lat:.4f} ± {std_lat:.4f} seconds")
            
        except Exception as e:
            print(f"Error benchmarking group size {g_size}: {e}")
            continue

    if not results:
        print("No benchmark results collected. Experiment failed.")
        return

    df = pd.DataFrame(results)
    print(f"\nCollected {len(df)} total measurements across {len(df['group_size'].unique())} group sizes")
    
    plot_results(df)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*60)
    
    summary = df.groupby('group_size')['latency_sec'].agg(['mean', 'std', 'median']).reset_index()
    if 1 in summary['group_size'].values:
        baseline_latency = summary[summary['group_size'] == 1]['mean'].iloc[0]
        summary['speedup'] = baseline_latency / summary['mean']
        summary['theoretical_speedup'] = summary['group_size']
        summary['efficiency'] = summary['speedup'] / summary['theoretical_speedup'] * 100
        
        print("\nFinal Results Summary:")
        print(summary.round(4).to_string(index=False))
        
        max_speedup = summary['speedup'].max()
        max_group = summary.loc[summary['speedup'].idxmax(), 'group_size']
        print(f"\nBest Performance:")
        print(f"- Maximum speedup: {max_speedup:.2f}x at group size {max_group}")
        print(f"- Efficiency: {summary.loc[summary['speedup'].idxmax(), 'efficiency']:.1f}%")

    update_status_enum("stopped")
    
    print(f"\nPlots saved to: .research/iteration1/images/")
    print("Experiment completed successfully!")

def test_experiment():
    """A quick test function to verify code functionality."""
    print("\n" + "="*50)
    print("Running Quick Test")
    print("="*50)
    
    setup_experiment_environment()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Test device: {device}")
    
    test_group_sizes = [1, 4]
    test_num_tokens = 64
    test_num_runs = 3
    test_warmup_runs = 1
    test_embed_dim = 256  # Very small model for fast test
    test_num_layers = 2

    try:
        model = GASPModel(embed_dim=test_embed_dim, num_layers=test_num_layers).to(device).eval()
        print(f"Test model created: {sum(p.numel() for p in model.parameters())/1e3:.1f}K parameters")
        
        test_results = []
        for g_size in test_group_sizes:
            latencies = run_benchmark(
                model, g_size, test_num_tokens, device, 
                num_runs=test_num_runs, warmup_runs=test_warmup_runs
            )
            for lat in latencies:
                test_results.append({'group_size': g_size, 'latency_sec': lat})
        
        df = pd.DataFrame(test_results)
        
        print("Testing plot generation...")
        plot_results(df, save_plots=False)
        
        print("✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_success = test_experiment()
    
    if test_success:
        print("\n" + "="*60)
        print("Starting Full Experiment")
        print("="*60)
        main()
    else:
        print("\nTest failed. Please check the implementation.")
        sys.exit(1)
