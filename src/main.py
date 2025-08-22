import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import yaml
import os
import argparse
from tqdm import tqdm

from preprocess import MockNYUv2Dataset
from train import MTLModel, MetaGradPolicy, train_epoch
from evaluate import evaluate_model

# ===============================================
# Plotting and Results Visualization
# ===============================================

def plot_training_curves(history, method, save_dir):
    """Plots and saves training loss curves."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history.get('segmentation', [])) + 1)
    if not epochs:
        print(f"No history to plot for {method}")
        return

    ax[0].plot(epochs, history['segmentation'], label='Segmentation Loss', color='b')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title(f'Segmentation Training Loss ({method})')
    ax[0].legend()

    ax[1].plot(epochs, history['depth'], label='Depth Loss', color='r')
    ax[1].set_xlabel('Epoch')
    ax[1].set_title(f'Depth Training Loss ({method})')
    ax[1].legend()
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"training_loss_{method.lower()}.pdf")
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved training curve plot to {filename}")
    plt.close()

def plot_pareto_fronts(results, save_dir):
    """Plots and saves the Pareto fronts for all methods."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    for i, method_name in enumerate(results.keys()):
        points = results[method_name]
        if not points: continue
        points.sort(key=lambda x: x[0])
        task1_perf = [p[0] for p in points]
        task2_perf = [p[1] for p in points]
        
        plt.plot(task1_perf, task2_perf, marker=markers[i % len(markers)], linestyle='-', label=method_name)
        plt.scatter(task1_perf, task2_perf, marker=markers[i % len(markers)])
    
    plt.xlabel('Segmentation Performance (Higher is better)')
    plt.ylabel('Depth Performance (Higher is better)')
    plt.title('Pareto Front Comparison on Mock NYUv2')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, "pareto_front_nyuv2.pdf")
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved Pareto front plot to {filename}")
    plt.close()

def print_hypervolume(results):
    """Calculates and prints the Hypervolume Indicator for each method."""
    print("\n--- Hypervolume Indicator --- (Higher is better)")
    
    all_points = np.array([p for method_res in results.values() for p in method_res])
    if len(all_points) == 0:
        print("No results to calculate hypervolume.")
        return
        
    ref_point = all_points.min(axis=0) - 0.1
    
    for method_name, points in results.items():
        if not points:
            print(f"{method_name}: 0.0")
            continue
        
        sorted_points = sorted(points, key=lambda p: p[0])
        hv = 0.0
        for x, y in sorted_points:
            if x > ref_point[0] and y > ref_point[1]:
                hv += (x - ref_point[0]) * (y - ref_point[1])
        print(f"{method_name}: {hv:.4f}")

# ===============================================
# Main Experiment Runner
# ===============================================

def run_experiment(config):
    """Main function to run the Pareto front benchmarking experiment."""
    print(f"Configuration: {config}\n")
    
    device = config['device'] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    full_results = defaultdict(list)
    image_save_dir = "../.research/iteration1/images"

    train_dataset = MockNYUv2Dataset(
        num_samples=config['batch_size'] * config['num_batches_per_epoch'],
        test_mode=config.get('test_mode', False),
        batch_size_for_test=config['batch_size']
    )
    val_dataset = MockNYUv2Dataset(
        num_samples=config['batch_size'] * 4,
        test_mode=config.get('test_mode', False),
        batch_size_for_test=config['batch_size']
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    for method_name in config['methods']:
        print(f'\n======= Running Method: {method_name} =======')
        method_training_history = defaultdict(list)

        for l_seg in tqdm(config['lambda_sweep'], desc=f"Lambda sweep for {method_name}"):
            l_depth = 1.0 - l_seg
            lambdas = [l_seg, l_depth]
            
            run_metrics_seg = []
            run_metrics_depth = []

            for seed in config['seeds']:
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                model = MTLModel(tasks=config['tasks']).to(device)
                policy = None
                optimizers = {}
                
                optimizer_model = optim.Adam(model.parameters(), lr=config['learning_rate_model'])
                optimizers['model'] = optimizer_model

                if method_name == 'MetaGrad':
                    policy = MetaGradPolicy(model, config['tasks'], rank=config['metagrad_rank']).to(device)
                    optimizer_policy = optim.Adam(policy.parameters(), lr=config['learning_rate_policy'])
                    optimizers['policy'] = optimizer_policy
                
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

                epoch_losses = defaultdict(list)
                for epoch in range(1, config['num_epochs'] + 1):
                    avg_losses, _ = train_epoch(model, optimizers, policy, train_loader, method_name, lambdas, device, config['tasks'])
                    if seed == config['seeds'][0]:
                         print(f'    Epoch {epoch}/{config["num_epochs"]} (Seed {seed}, Lambda {l_seg:.1f}) - Seg Loss: {avg_losses["segmentation"]:.4f}, Depth Loss: {avg_losses["depth"]:.4f}')
                    for task, loss in avg_losses.items():
                        epoch_losses[task].append(loss)

                if abs(l_seg - config['lambda_sweep'][0]) < 1e-5: 
                    method_training_history = epoch_losses

                metrics = evaluate_model(model, val_loader, device)
                run_metrics_seg.append(metrics['segmentation_perf'])
                run_metrics_depth.append(metrics['depth_perf'])
            
            avg_seg_perf = np.mean(run_metrics_seg)
            avg_depth_perf = np.mean(run_metrics_depth)
            full_results[method_name].append((avg_seg_perf, avg_depth_perf))
        
        plot_training_curves(method_training_history, method_name, image_save_dir)
    
    print("\nExperiment finished. Generating plots and results...")
    plot_pareto_fronts(full_results, image_save_dir)
    print_hypervolume(full_results)

def main():
    parser = argparse.ArgumentParser(description="Run MTL Pareto Front Experiment.")
    parser.add_argument('--test', action='store_true', help="Run in a quick test mode.")
    parser.add_argument('--config', type=str, default='../config/config.yaml', help="Path to config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.test:
        print("--- RUNNING IN TEST MODE ---")
        config['methods'] = ['Uniform', 'MetaGrad']
        config['seeds'] = [42]
        config['lambda_sweep'] = [0.5]
        config['num_epochs'] = 1
        config['batch_size'] = 4
        config['num_batches_per_epoch'] = 2
        config['test_mode'] = True
    
    print("=============================================")
    print("         STARTING EXPERIMENT          ")
    print("=============================================")
    run_experiment(config)

if __name__ == '__main__':
    main()
