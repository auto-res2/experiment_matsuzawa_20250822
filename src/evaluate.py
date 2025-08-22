"""
Evaluation script for SEEDS experiments
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import time
from scipy.stats import ks_2samp
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.seeds_config import config
from src.models import ImagePtheta, ImageSurrogate, SequencePtheta, SequenceSurrogate
from src.diffusion_utils import BetaSchedule
from src.seeds_sampler import SEEDSSampler, compare_samplers
from src.preprocess import load_datasets

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_trained_models(device: str = 'cpu'):
    """Load trained models and kappa values"""
    print("Loading trained models...")
    
    image_ptheta = ImagePtheta(config.K)
    image_surrogate = ImageSurrogate(config.K)
    seq_ptheta = SequencePtheta(config.K)
    seq_surrogate = SequenceSurrogate(config.K)
    
    models_dir = config.models_dir
    
    image_ptheta.load_state_dict(torch.load(os.path.join(models_dir, 'image_ptheta.pt'), map_location=device, weights_only=False))
    image_surrogate.load_state_dict(torch.load(os.path.join(models_dir, 'image_surrogate.pt'), map_location=device, weights_only=False))
    seq_ptheta.load_state_dict(torch.load(os.path.join(models_dir, 'seq_ptheta.pt'), map_location=device, weights_only=False))
    seq_surrogate.load_state_dict(torch.load(os.path.join(models_dir, 'seq_surrogate.pt'), map_location=device, weights_only=False))
    
    kappa_dict = torch.load(os.path.join(models_dir, 'kappa_values.pt'), map_location=device, weights_only=False)
    
    models = {
        'image_ptheta': image_ptheta,
        'image_surrogate': image_surrogate,
        'seq_ptheta': seq_ptheta,
        'seq_surrogate': seq_surrogate
    }
    
    return models, kappa_dict

def evaluate_sampling_efficiency(models: Dict, kappa_dict: Dict, datasets: Dict, 
                                device: str = 'cpu') -> Dict[str, Any]:
    """Evaluate sampling efficiency of SEEDS vs baselines"""
    print("Evaluating sampling efficiency...")
    
    sched = BetaSchedule(config.K, config.beta_params)
    results = {}
    
    print("Running image sampling experiments...")
    image_test_data = datasets['image_test']
    
    n_test = min(5, len(image_test_data))
    test_indices = torch.randperm(len(image_test_data))[:n_test]
    
    image_results = []
    for i, idx in enumerate(test_indices):
        x_init = image_test_data[idx].to(device)
        
        t_start = torch.tensor(0.8, device=device)
        from src.diffusion_utils import forward_corrupt
        x_noisy = forward_corrupt(x_init.unsqueeze(0), t_start, config.K, sched).squeeze(0)
        
        comparison = compare_samplers(
            models['image_ptheta'], models['image_surrogate'], 
            kappa_dict['image_kappa'], x_noisy, config.K, sched, device
        )
        
        image_results.append(comparison)
        print(f"Image {i+1}/{n_test} completed")
    
    results['image'] = image_results
    
    print("Running sequence sampling experiments...")
    seq_test_data = datasets['seq_test']
    
    n_test = min(5, len(seq_test_data))
    test_indices = torch.randperm(len(seq_test_data))[:n_test]
    
    seq_results = []
    for i, idx in enumerate(test_indices):
        x_init = seq_test_data[idx].to(device)
        
        t_start = torch.tensor(0.8, device=device)
        x_noisy = forward_corrupt(x_init.unsqueeze(0), t_start, config.K, sched).squeeze(0)
        
        comparison = compare_samplers(
            models['seq_ptheta'], models['seq_surrogate'], 
            kappa_dict['seq_kappa'], x_noisy, config.K, sched, device
        )
        
        seq_results.append(comparison)
        print(f"Sequence {i+1}/{n_test} completed")
    
    results['sequence'] = seq_results
    
    return results

def compute_quality_metrics(original: torch.Tensor, generated: torch.Tensor) -> Dict[str, float]:
    """Compute quality metrics between original and generated samples"""
    
    orig_np = original.cpu().numpy().flatten()
    gen_np = generated.cpu().numpy().flatten()
    
    ks_stat, ks_pval = ks_2samp(orig_np, gen_np)
    
    mse = np.mean((orig_np - gen_np) ** 2)
    
    hamming = np.mean(orig_np != gen_np)
    
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'mse': mse,
        'hamming_distance': hamming
    }

def create_efficiency_plots(results: Dict[str, Any], save_dir: str):
    """Create efficiency comparison plots"""
    print("Creating efficiency plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    methods = ['seeds_exact', 'seeds_budgeted', 'tau_leaping']
    method_names = ['SEEDS (Exact)', 'SEEDS (Budgeted)', 'Tau-leaping']
    
    for data_type in ['image', 'sequence']:
        if data_type not in results:
            continue
            
        data_results = results[data_type]
        
        nfe_heavy = {method: [] for method in methods}
        nfe_surrogate = {method: [] for method in methods}
        wall_time = {method: [] for method in methods}
        n_events = {method: [] for method in methods}
        acceptance_rate = {method: [] for method in methods}
        
        for result in data_results:
            for method in methods:
                if method in result:
                    stats = result[method]['stats']
                    nfe_heavy[method].append(stats.nfe_heavy)
                    nfe_surrogate[method].append(stats.nfe_surrogate)
                    wall_time[method].append(stats.wall_time)
                    n_events[method].append(stats.n_events)
                    acceptance_rate[method].append(stats.acceptance_rate)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'SEEDS Efficiency Analysis - {data_type.title()} Data', fontsize=16)
        
        ax = axes[0, 0]
        means = [float(np.mean(nfe_heavy[method])) for method in methods]
        stds = [float(np.std(nfe_heavy[method])) for method in methods]
        bars = ax.bar(method_names, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_ylabel('Heavy NFE (p_theta calls)')
        ax.set_title('Model Evaluations')
        ax.tick_params(axis='x', rotation=45)
        
        max_std = max(stds) if stds else 0.0
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_std/20,
                   f'{mean:.1f}', ha='center', va='bottom')
        
        ax = axes[0, 1]
        means = [float(np.mean(wall_time[method])) for method in methods]
        stds = [float(np.std(wall_time[method])) for method in methods]
        bars = ax.bar(method_names, means, yerr=stds, capsize=5, alpha=0.7, color='orange')
        ax.set_ylabel('Wall Time (seconds)')
        ax.set_title('Runtime Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        max_std = max(stds) if stds else 0.0
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_std/20,
                   f'{mean:.3f}', ha='center', va='bottom')
        
        ax = axes[0, 2]
        means = [float(np.mean(n_events[method])) for method in methods]
        stds = [float(np.std(n_events[method])) for method in methods]
        bars = ax.bar(method_names, means, yerr=stds, capsize=5, alpha=0.7, color='green')
        ax.set_ylabel('Number of Events')
        ax.set_title('Sampling Events')
        ax.tick_params(axis='x', rotation=45)
        
        max_std = max(stds) if stds else 0.0
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_std/20,
                   f'{mean:.1f}', ha='center', va='bottom')
        
        ax = axes[1, 0]
        means = [np.mean(acceptance_rate[method]) for method in methods if acceptance_rate[method]]
        method_subset = [method for method in methods if acceptance_rate[method]]
        name_subset = [method_names[i] for i, method in enumerate(methods) if acceptance_rate[method]]
        stds = [np.std(acceptance_rate[method]) for method in method_subset]
        
        if means:
            bars = ax.bar(name_subset, means, yerr=stds, capsize=5, alpha=0.7, color='red')
            ax.set_ylabel('Acceptance Rate')
            ax.set_title('Event Acceptance Rate')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1)
            
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{mean:.3f}', ha='center', va='bottom')
        
        ax = axes[1, 1]
        if nfe_heavy['tau_leaping']:
            baseline_nfe = np.mean(nfe_heavy['tau_leaping'])
            ratios = []
            ratio_names = []
            for i, method in enumerate(['seeds_exact', 'seeds_budgeted']):
                if nfe_heavy[method]:
                    ratio = baseline_nfe / np.mean(nfe_heavy[method])
                    ratios.append(ratio)
                    ratio_names.append(method_names[i])
            
            if ratios:
                bars = ax.bar(ratio_names, ratios, alpha=0.7, color='purple')
                ax.set_ylabel('Speedup Factor')
                ax.set_title('NFE Reduction vs Tau-leaping')
                ax.tick_params(axis='x', rotation=45)
                ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
                
                for bar, ratio in zip(bars, ratios):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{ratio:.1f}x', ha='center', va='bottom')
        
        ax = axes[1, 2]
        total_nfe = []
        for method in methods:
            if nfe_heavy[method] and nfe_surrogate[method]:
                total = np.mean(nfe_heavy[method]) + np.mean(nfe_surrogate[method])
                total_nfe.append(total)
            elif nfe_heavy[method]:
                total_nfe.append(np.mean(nfe_heavy[method]))
            else:
                total_nfe.append(0)
        
        bars = ax.bar(method_names, total_nfe, alpha=0.7, color='brown')
        ax.set_ylabel('Total NFE')
        ax.set_title('Total Function Evaluations')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, total in zip(bars, total_nfe):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(total_nfe) if total_nfe else 0)/50,
                   f'{total:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        filename = f'efficiency_comparison_{data_type}.pdf'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved efficiency plot: {filepath}")

def create_sample_visualization(results: Dict[str, Any], datasets: Dict, save_dir: str):
    """Create visualizations of generated samples"""
    print("Creating sample visualizations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if 'image' in results and results['image']:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle('Image Sample Comparison', fontsize=16)
        
        result = results['image'][0]
        methods = ['seeds_exact', 'seeds_budgeted', 'tau_leaping']
        method_names = ['SEEDS (Exact)', 'SEEDS (Budgeted)', 'Tau-leaping']
        
        original = datasets['image_test'][0].cpu().numpy()
        
        for i, (method, name) in enumerate(zip(methods, method_names)):
            if method in result:
                sample = result[method]['sample'].cpu().numpy()
                
                axes[i, 0].imshow(original, cmap='tab10', vmin=0, vmax=config.K-1)
                axes[i, 0].set_title('Original' if i == 0 else '')
                axes[i, 0].set_ylabel(name)
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(sample, cmap='tab10', vmin=0, vmax=config.K-1)
                axes[i, 1].set_title('Generated' if i == 0 else '')
                axes[i, 1].axis('off')
                
                diff = (original != sample).astype(float)
                axes[i, 2].imshow(diff, cmap='Reds', vmin=0, vmax=1)
                axes[i, 2].set_title('Differences' if i == 0 else '')
                axes[i, 2].axis('off')
                
                stats = result[method]['stats']
                stats_text = f"NFE: {stats.nfe_heavy}\nTime: {stats.wall_time:.3f}s\nEvents: {stats.n_events}"
                axes[i, 3].text(0.1, 0.5, stats_text, transform=axes[i, 3].transAxes, 
                               fontsize=10, verticalalignment='center')
                axes[i, 3].set_title('Statistics' if i == 0 else '')
                axes[i, 3].axis('off')
        
        plt.tight_layout()
        filepath = os.path.join(save_dir, 'image_samples.pdf')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved image samples: {filepath}")
    
    if 'sequence' in results and results['sequence']:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle('Sequence Sample Comparison', fontsize=16)
        
        result = results['sequence'][0]
        methods = ['seeds_exact', 'seeds_budgeted', 'tau_leaping']
        method_names = ['SEEDS (Exact)', 'SEEDS (Budgeted)', 'Tau-leaping']
        
        original = datasets['seq_test'][0].cpu().numpy()
        
        for i, (method, name) in enumerate(zip(methods, method_names)):
            if method in result:
                sample = result[method]['sample'].cpu().numpy()
                
                x = np.arange(len(original))
                axes[i].plot(x, original, 'o-', label='Original', alpha=0.7, markersize=4)
                axes[i].plot(x, sample, 's-', label='Generated', alpha=0.7, markersize=4)
                axes[i].set_ylabel('State')
                axes[i].set_title(name)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylim(-0.5, config.K - 0.5)
        
        axes[-1].set_xlabel('Position')
        
        plt.tight_layout()
        filepath = os.path.join(save_dir, 'sequence_samples.pdf')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved sequence samples: {filepath}")

def create_summary_report(results: Dict[str, Any], save_dir: str):
    """Create summary statistics report"""
    print("Creating summary report...")
    
    summary_stats = {}
    
    for data_type in ['image', 'sequence']:
        if data_type not in results:
            continue
            
        data_results = results[data_type]
        methods = ['seeds_exact', 'seeds_budgeted', 'tau_leaping']
        
        summary_stats[data_type] = {}
        
        for method in methods:
            nfe_heavy = []
            wall_time = []
            n_events = []
            
            for result in data_results:
                if method in result:
                    stats = result[method]['stats']
                    nfe_heavy.append(stats.nfe_heavy)
                    wall_time.append(stats.wall_time)
                    n_events.append(stats.n_events)
            
            if nfe_heavy:
                summary_stats[data_type][method] = {
                    'nfe_heavy_mean': np.mean(nfe_heavy),
                    'nfe_heavy_std': np.std(nfe_heavy),
                    'wall_time_mean': np.mean(wall_time),
                    'wall_time_std': np.std(wall_time),
                    'n_events_mean': np.mean(n_events),
                    'n_events_std': np.std(n_events)
                }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Data Type', 'Method', 'NFE Heavy', 'Wall Time (s)', 'Events']
    
    for data_type, methods_data in summary_stats.items():
        for method, stats in methods_data.items():
            method_name = {
                'seeds_exact': 'SEEDS (Exact)',
                'seeds_budgeted': 'SEEDS (Budgeted)',
                'tau_leaping': 'Tau-leaping'
            }[method]
            
            row = [
                data_type.title(),
                method_name,
                f"{stats['nfe_heavy_mean']:.1f} ± {stats['nfe_heavy_std']:.1f}",
                f"{stats['wall_time_mean']:.3f} ± {stats['wall_time_std']:.3f}",
                f"{stats['n_events_mean']:.1f} ± {stats['n_events_std']:.1f}"
            ]
            table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f1f1f2')
    
    plt.title('SEEDS Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    filepath = os.path.join(save_dir, 'summary_table.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary table: {filepath}")
    
    return summary_stats

def main():
    """Main evaluation function"""
    set_seed(config.seed)
    
    device = config.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    models, kappa_dict = load_trained_models(device)
    datasets = load_datasets()
    
    print(f"Loaded models with kappa values: {kappa_dict}")
    
    results = evaluate_sampling_efficiency(models, kappa_dict, datasets, device)
    
    save_dir = config.results_dir
    create_efficiency_plots(results, save_dir)
    create_sample_visualization(results, datasets, save_dir)
    summary_stats = create_summary_report(results, save_dir)
    
    print("Evaluation completed successfully!")
    print(f"Results saved to: {save_dir}")
    
    return results, summary_stats

if __name__ == "__main__":
    results, summary = main()
