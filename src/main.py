import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

try:
    from .train import GCNEncoder, ProjectionHead, InfoNCE_GCL, ASAR_GCL, train_model
    from .evaluate import evaluate_model_multiple_runs
    from .preprocess import prepare_datasets
except ImportError:
    from train import GCNEncoder, ProjectionHead, InfoNCE_GCL, ASAR_GCL, train_model
    from evaluate import evaluate_model_multiple_runs
    from preprocess import prepare_datasets

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_PRETRAIN_EPOCHS = 20  # Reduced for testing on T4
NUM_EVAL_RUNS = 3  # Reduced for faster testing

def create_model(model_name, num_features):
    """Create a model based on the model name."""
    encoder = GCNEncoder(num_features, 256, 256)
    projection_head = ProjectionHead(256, 256, 128)
    
    if model_name == "GRACE":
        return InfoNCE_GCL(encoder, projection_head)
    elif model_name == "GCA":
        return InfoNCE_GCL(encoder, projection_head)
    elif model_name == "ASAR (Fixed Repulsion)":
        return ASAR_GCL(encoder, projection_head, dynamic_lambda=False, dynamic_margin=True, fixed_lambda=1.0)
    elif model_name == "ASAR (Fixed Margin)":
        return ASAR_GCL(encoder, projection_head, dynamic_lambda=True, dynamic_margin=False, fixed_margin=0.1)
    elif model_name == "ASAR (Full)":
        return ASAR_GCL(encoder, projection_head, dynamic_lambda=True, dynamic_margin=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def plot_results(results_df, dataset_name, output_dir):
    """Generate and save bar plots for metrics."""
    print(f"Generating plots for {dataset_name}...")
    
    metrics = ['Test Accuracy', 'Training Time (s)', 'Peak GPU Memory (GB)']
    filenames = ['accuracy', 'training_time', 'peak_memory']

    for metric, filename in zip(metrics, filenames):
        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")
        bar_plot = sns.barplot(
            data=results_df,
            x='Model',
            y=metric,
            palette='viridis',
            hue='Model',
            dodge=False,
            legend=False
        )
        plt.title(f'{metric} Comparison on {dataset_name}', fontsize=16)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.xticks(rotation=15, ha='right')
        
        for i, (model, value) in enumerate(zip(results_df['Model'], results_df[metric])):
            if not np.isinf(value) and not np.isnan(value) and value != 0:
                plt.text(i, value + 0.01 * max(results_df[metric]), f'{value:.3f}', 
                        ha='center', va='bottom', fontsize=10)
        
        output_path = os.path.join(output_dir, f'{filename}_{dataset_name.replace("-", "_")}.pdf')
        plt.savefig(output_path, bbox_inches='tight', format='pdf', dpi=300)
        print(f"Saved plot to {output_path}")
        plt.close()

def run_experiment_on_dataset(dataset_name, dataset_info, output_dir):
    """Run the complete experiment on a single dataset."""
    if dataset_info is None:
        print(f"Skipping {dataset_name} - dataset not available")
        return None
    
    data = dataset_info['data']
    loader = dataset_info['loader']
    split_idx = dataset_info['split_idx']
    
    models_to_test = [
        "GRACE",
        "GCA", 
        "ASAR (Fixed Repulsion)",
        "ASAR (Fixed Margin)",
        "ASAR (Full)"
    ]

    dataset_results = []
    print(f"\n--- Starting Experiment on {dataset_name} ---")
    
    for model_name in models_to_test:
        torch.manual_seed(42)
        np.random.seed(42)
        print(f"\n>>> Training model: {model_name} on {dataset_name}...")
        
        model = create_model(model_name, data.num_features).to(DEVICE)
        
        training_time, peak_mem_gb = train_model(
            model, loader, NUM_PRETRAIN_EPOCHS, DEVICE
        )
        
        mean_acc, std_acc, accuracies = evaluate_model_multiple_runs(
            model.encoder, data, split_idx, DEVICE, NUM_EVAL_RUNS
        )
        
        dataset_results.append({
            'Model': model_name,
            'Test Accuracy': mean_acc,
            'Accuracy Std': std_acc,
            'Training Time (s)': training_time,
            'Peak GPU Memory (GB)': peak_mem_gb
        })
        
        print(f"Results for {model_name}: Acc={mean_acc:.4f}±{std_acc:.4f}, "
              f"Time={training_time:.2f}s, Memory={peak_mem_gb:.3f}GB")

    dataset_results.append({
        'Model': 'ContraNorm',
        'Test Accuracy': 0.0,
        'Accuracy Std': 0.0,
        'Training Time (s)': float('inf'),
        'Peak GPU Memory (GB)': float('inf')
    })

    results_df = pd.DataFrame(dataset_results)
    plot_results(results_df, dataset_name, output_dir)
    
    return results_df

def save_results_summary(all_results, output_dir):
    """Save a summary of all results."""
    summary_path = os.path.join(output_dir, 'experiment_summary.json')
    
    summary = {
        'experiment_date': datetime.now().isoformat(),
        'device': str(DEVICE),
        'num_pretrain_epochs': NUM_PRETRAIN_EPOCHS,
        'num_eval_runs': NUM_EVAL_RUNS,
        'results': {}
    }
    
    for dataset_name, results_df in all_results.items():
        if results_df is not None:
            summary['results'][dataset_name] = results_df.to_dict('records')
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Experiment summary saved to {summary_path}")

def main():
    """Main function to execute the large-scale performance and scalability experiment."""
    print(f"Running ASAR experiment on device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    output_dir = '.research/iteration1/images'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Preparing Datasets ===")
    datasets = prepare_datasets(DEVICE)
    
    all_results = {}
    
    for dataset_name in ["ogbn-arxiv", "ogbn-products"]:
        if datasets[dataset_name] is not None:
            results_df = run_experiment_on_dataset(
                dataset_name, datasets[dataset_name], output_dir
            )
            all_results[dataset_name] = results_df
        else:
            all_results[dataset_name] = None
    
    save_results_summary(all_results, output_dir)
    
    print("\n=== Experiment Complete ===")
    print("All results and plots have been saved to .research/iteration1/images/")
    
    for dataset_name, results_df in all_results.items():
        if results_df is not None:
            print(f"\n{dataset_name} Results:")
            print(results_df[['Model', 'Test Accuracy', 'Training Time (s)', 'Peak GPU Memory (GB)']].to_string(index=False))

if __name__ == "__main__":
    main()
