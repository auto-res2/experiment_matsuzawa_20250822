"""
Evaluation and Benchmarking Module for GASP Latency Experiment

This module contains functions for running latency benchmarks and generating
analysis plots for the GASP model performance evaluation.
"""

import torch
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path

def run_benchmark(model, group_size, num_tokens, device, num_runs=100, warmup_runs=20):
    """
    Runs a timed benchmark for a given model configuration.
    
    Args:
        model: GASP model instance
        group_size: Group size for generation
        num_tokens: Total number of tokens to generate
        device: Device to run on ('cuda' or 'cpu')
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        List of latency measurements in seconds
    """
    latencies = []
    
    if device == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_amp = True
    else:
        dtype = torch.float32
        use_amp = False

    with torch.no_grad():
        if use_amp:
            autocast_context = torch.cuda.amp.autocast(dtype=dtype)
        else:
            autocast_context = torch.no_grad()  # Dummy context for CPU
            
        with autocast_context:
            print(f"Warming up for G={group_size} ({warmup_runs} runs)...")
            for _ in range(warmup_runs):
                try:
                    _ = model.generate(num_tokens=num_tokens, group_size=group_size)
                except Exception as e:
                    print(f"Warmup failed: {e}")
                    raise
            
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            print(f"Benchmarking G={group_size} ({num_runs} runs)...")
            for run_idx in tqdm(range(num_runs), desc=f'G={group_size}'):
                try:
                    if device == 'cuda':
                        torch.cuda.synchronize(device)
                    
                    start_time = time.perf_counter()
                    
                    _ = model.generate(num_tokens=num_tokens, group_size=group_size)
                    
                    if device == 'cuda':
                        torch.cuda.synchronize(device)
                    
                    end_time = time.perf_counter()
                    latency = end_time - start_time
                    latencies.append(latency)
                    
                except Exception as e:
                    print(f"Run {run_idx} failed: {e}")
                    continue

    if not latencies:
        raise RuntimeError(f"No successful runs for group size {group_size}")
    
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    median_lat = np.median(latencies)
    
    print(f"Group {group_size} Results:")
    print(f"  Mean: {mean_lat:.4f}s ± {std_lat:.4f}s")
    print(f"  Median: {median_lat:.4f}s")
    print(f"  Min: {min(latencies):.4f}s, Max: {max(latencies):.4f}s")
    
    return latencies

def plot_results(df, save_plots=True):
    """
    Analyzes the benchmark data and generates plots.
    
    Args:
        df: DataFrame with columns ['group_size', 'latency_sec']
        save_plots: Whether to save plots to files
    """
    print("\n" + "="*50)
    print("Analysis and Visualization")
    print("="*50)
    
    if df.empty:
        print("DataFrame is empty, skipping plotting.")
        return

    plt.style.use('default')
    sns.set_palette("viridis")
    
    if save_plots:
        output_dir = Path(__file__).parent.parent / ".research" / "iteration1" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    group_sizes = sorted(df['group_size'].unique())
    
    plt.figure(figsize=(10, 6))
    
    stats_df = df.groupby('group_size')['latency_sec'].agg(['mean', 'std', 'count']).reset_index()
    stats_df['sem'] = stats_df['std'] / np.sqrt(stats_df['count'])  # Standard error of mean
    
    plt.errorbar(stats_df['group_size'], stats_df['mean'], 
                yerr=stats_df['sem'], marker='o', linewidth=2, 
                markersize=8, capsize=5, capthick=2)
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Group Size (G)', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Latency per Image (seconds)', fontsize=14, fontweight='bold')
    plt.title('GASP Model: Inference Latency vs Group Size', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(group_sizes, labels=[str(g) for g in group_sizes])
    
    for _, row in stats_df.iterrows():
        plt.annotate(f'{row["mean"]:.3f}s', 
                    (row['group_size'], row['mean']),
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots and output_dir:
        filename = output_dir / "inference_latency_vs_group_size.pdf"
        plt.savefig(filename, dpi=300, bbox_inches="tight", format='pdf')
        print(f"✓ Saved: {filename}")
    
    plt.close()

    if 1 in stats_df['group_size'].values:
        baseline_latency = stats_df[stats_df['group_size'] == 1]['mean'].iloc[0]
        stats_df['speedup'] = baseline_latency / stats_df['mean']
        stats_df['theoretical_speedup'] = stats_df['group_size']
        stats_df['efficiency'] = (stats_df['speedup'] / stats_df['theoretical_speedup']) * 100
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(stats_df['group_size'], stats_df['speedup'], 
                'o-', linewidth=3, markersize=8, label='Measured Speedup', color='#1f77b4')
        
        plt.plot(group_sizes, group_sizes, '--', linewidth=2, 
                label='Theoretical Linear Speedup', color='red', alpha=0.7)
        
        plt.xscale('log', base=2)
        plt.xlabel('Group Size (G)', fontsize=14, fontweight='bold')
        plt.ylabel('Speedup Factor (relative to G=1)', fontsize=14, fontweight='bold')
        plt.title('GASP Model: Inference Speedup vs Group Size', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(group_sizes, labels=[str(g) for g in group_sizes])
        plt.legend(fontsize=12)
        
        for _, row in stats_df.iterrows():
            if row['group_size'] > 1:
                plt.annotate(f'{row["efficiency"]:.0f}%', 
                            (row['group_size'], row['speedup']),
                            textcoords="offset points", xytext=(0,10), ha='center',
                            fontsize=10, fontweight='bold', color='darkgreen')
        
        plt.tight_layout()
        
        if save_plots and output_dir:
            filename = output_dir / "inference_speedup_vs_group_size.pdf"
            plt.savefig(filename, dpi=300, bbox_inches="tight", format='pdf')
            print(f"✓ Saved: {filename}")
        
        plt.close()
        
        plt.figure(figsize=(10, 6))
        
        efficiency_data = stats_df[stats_df['group_size'] > 1].copy()
        bars = plt.bar(range(len(efficiency_data)), efficiency_data['efficiency'], 
                      color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=2)
        
        plt.axhline(y=100, color='red', linestyle='--', linewidth=2, 
                   label='Perfect Efficiency (100%)')
        plt.axhline(y=80, color='orange', linestyle=':', linewidth=2, 
                   label='Good Efficiency (80%)')
        
        plt.xlabel('Group Size', fontsize=14, fontweight='bold')
        plt.ylabel('Efficiency (%)', fontsize=14, fontweight='bold')
        plt.title('GASP Model: Parallel Efficiency by Group Size', fontsize=16, fontweight='bold')
        plt.xticks(range(len(efficiency_data)), 
                  [f'G={int(g)}' for g in efficiency_data['group_size']])
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=12)
        
        for i, (bar, eff) in enumerate(zip(bars, efficiency_data['efficiency'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{eff:.1f}%', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
        
        plt.ylim(0, 110)
        plt.tight_layout()
        
        if save_plots and output_dir:
            filename = output_dir / "parallel_efficiency_analysis.pdf"
            plt.savefig(filename, dpi=300, bbox_inches="tight", format='pdf')
            print(f"✓ Saved: {filename}")
        
        plt.close()

    print("\n" + "="*60)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*60)
    
    summary_table = stats_df.copy()
    summary_table = summary_table.round(4)
    
    if 'speedup' in summary_table.columns:
        print("\nLatency and Speedup Analysis:")
        print(summary_table[['group_size', 'mean', 'std', 'speedup', 'efficiency']].to_string(index=False))
        
        max_speedup_idx = summary_table['speedup'].idxmax()
        max_speedup = summary_table.loc[max_speedup_idx, 'speedup']
        max_speedup_group = summary_table.loc[max_speedup_idx, 'group_size']
        max_efficiency = summary_table.loc[max_speedup_idx, 'efficiency']
        
        print(f"\nKey Findings:")
        print(f"• Maximum speedup: {max_speedup:.2f}x at group size {int(max_speedup_group)}")
        print(f"• Best efficiency: {max_efficiency:.1f}% at group size {int(max_speedup_group)}")
        print(f"• Baseline latency (G=1): {baseline_latency:.4f} seconds")
        
        total_theoretical = sum(group_sizes)
        total_actual = sum(summary_table['speedup'])
        overall_efficiency = (total_actual / total_theoretical) * 100
        print(f"• Overall parallel efficiency: {overall_efficiency:.1f}%")
        
    else:
        print("\nLatency Analysis:")
        print(summary_table[['group_size', 'mean', 'std', 'count']].to_string(index=False))
    
    print(f"\nTotal measurements collected: {len(df)}")
    print(f"Group sizes tested: {group_sizes}")
    
    if save_plots and output_dir:
        print(f"\nAll plots saved to: {output_dir}")
        print("✓ Analysis complete!")

def calculate_memory_usage(model, group_size, num_tokens, device):
    """
    Estimate memory usage for different group sizes.
    
    Args:
        model: GASP model instance
        group_size: Group size for generation
        num_tokens: Total number of tokens
        device: Device ('cuda' or 'cpu')
        
    Returns:
        Dictionary with memory statistics
    """
    if device != 'cuda':
        return {'peak_memory_mb': 0, 'allocated_memory_mb': 0}
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        try:
            _ = model.generate(num_tokens=num_tokens, group_size=group_size)
            
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            current_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)
            
            return {
                'peak_memory_mb': peak_memory,
                'allocated_memory_mb': current_memory
            }
        except Exception as e:
            print(f"Memory measurement failed: {e}")
            return {'peak_memory_mb': 0, 'allocated_memory_mb': 0}
