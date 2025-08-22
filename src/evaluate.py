import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm
import os

def evaluate_model(model, loader, cfg, device):
    model.eval()
    model.to(device)
    
    all_latencies = []
    all_gflops = []
    all_accuracies = []

    desc = f"Evaluating {cfg['name']}"
    if 'budget' in cfg: desc += f" @ budget={cfg['budget']}"
    if 'threshold' in cfg: desc += f" @ thr={cfg['threshold']:.2f}"
        
    with torch.no_grad():
        for features, _, difficulty in tqdm(loader, desc=desc, leave=False):
            features, difficulty = features.to(device), difficulty.to(device)

            # --- Cost Measurement ---
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            if cfg['type'] == 'metabapl' or cfg['type'] == 'metabapl_interp':
                budget_tensor = torch.tensor(cfg['budget'], dtype=torch.float32).to(device)
                accuracy_score, gflops = model(features, difficulty, budget=budget_tensor)
            elif cfg['type'] == 'confidence':
                accuracy_score, gflops = model(features, difficulty, confidence_threshold=cfg['threshold'])
            else: # Static or Policy Zoo
                accuracy_score, gflops = model(features, difficulty)

            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Collect batch-level results
            all_latencies.append((end_time - start_time) * 1000 / len(features))
            all_gflops.append(gflops)
            # Accuracy is simulated as a score. We cap it at 0.95.
            all_accuracies.append(torch.clamp(accuracy_score, 0, 0.95).item())
    
    # Aggregate results
    results = {
        'model': cfg['name'],
        'type': cfg['type'],
        'accuracy': np.mean(all_accuracies) * 100, # As percentage
        'latency_ms': np.mean(all_latencies),
        'gflops': np.mean(all_gflops),
        'latency_std': np.std(all_latencies),
        'gflops_std': np.std(all_gflops),
        'per_batch_gflops': all_gflops
    }
    if 'budget' in cfg:
        results['target_latency_ms'] = cfg['budget'][0]
        results['target_gflops'] = cfg['budget'][1]
    if 'threshold' in cfg:
        results['threshold'] = cfg['threshold']
        
    return results

def plot_pareto_frontier(df, figure_dir):
    print("\nGenerating Pareto Frontier plots...")
    for cost_metric in ['latency_ms', 'gflops']:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.lineplot(data=df[df['type']=='metabapl'], x=cost_metric, y='accuracy', 
                     marker='o', label='Meta-BAPL (Ours)', ax=ax, sort=False, color='red', zorder=5)
        sns.scatterplot(data=df[df['type']=='metabapl_interp'], x=cost_metric, y='accuracy', 
                        marker='x', s=100, label='Meta-BAPL (Interpolated)', ax=ax, color='darkred', zorder=5)
        
        sns.lineplot(data=df[df['type']=='confidence'], x=cost_metric, y='accuracy', 
                     marker='s', label='Confidence-based', ax=ax, color='blue')
        
        sns.scatterplot(data=df[df['type']=='model_zoo'], x=cost_metric, y='accuracy', 
                        marker='^', s=200, label='Model Zoo', ax=ax, color='green')

        sns.scatterplot(data=df[df['type']=='policy_zoo'], x=cost_metric, y='accuracy', 
                        marker='D', s=150, label='Policy Zoo', ax=ax, color='purple')

        ax.set_title(f'Accuracy vs. {cost_metric.replace("_", " ").title()}', fontsize=16)
        ax.set_xlabel(f'{cost_metric.replace("_", " ").title()}', fontsize=12)
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax.legend(title='Method')
        ax.grid(True, which='both', linestyle='--')
        
        filename = os.path.join(figure_dir, f'accuracy_vs_{cost_metric}.pdf')
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.close(fig)

def plot_budget_adherence(df, figure_dir):
    print("\nGenerating Budget Adherence plots...")
    metabapl_df = df[df['type'].str.contains('metabapl')].copy()
    
    for budget_type in ['gflops', 'latency_ms']:
        target_col = f'target_{budget_type}'
        measured_col = budget_type
        
        if target_col not in metabapl_df.columns:
            continue

        mape = np.mean(np.abs((metabapl_df[measured_col] - metabapl_df[target_col]) / metabapl_df[target_col])) * 100
        print(f"MAPE for {budget_type}: {mape:.2f}%")

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(7, 7))
        
        max_val = max(metabapl_df[target_col].max(), metabapl_df[measured_col].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', label='Perfect Adherence (y=x)')
        
        sns.scatterplot(data=metabapl_df, x=target_col, y=measured_col, ax=ax, label='Measured Performance')
        
        ax.set_title(f'Budget Adherence for {budget_type.replace("_", " ").title()}', fontsize=16)
        ax.set_xlabel(f'Requested Budget', fontsize=12)
        ax.set_ylabel(f'Measured Mean Cost', fontsize=12)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        
        filename = os.path.join(figure_dir, f'budget_adherence_{budget_type}.pdf')
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.close(fig)

def plot_cost_consistency(df, figure_dir):
    print("\nGenerating Cost Consistency plot...")
    metabapl_df = df[df['type'].str.contains('metabapl')]
    
    # Select a few budgets to plot histograms for
    budgets_to_plot = metabapl_df['target_gflops'].quantile([0.25, 0.5, 0.75]).unique()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for target_gflops in budgets_to_plot:
        data = metabapl_df[metabapl_df['target_gflops'] == target_gflops]
        if not data.empty:
            per_batch_data = data.iloc[0]['per_batch_gflops']
            sns.histplot(per_batch_data, ax=ax, kde=True, label=f'Budget: {target_gflops:.1f} GFLOPs', bins=10)
    
    ax.set_title('Per-Batch Cost Consistency for Meta-BAPL', fontsize=16)
    ax.set_xlabel('GFLOPs per Batch', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()

    filename = os.path.join(figure_dir, 'cost_consistency_gflops.pdf')
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.close(fig)
