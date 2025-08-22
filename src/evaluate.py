import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


def evaluate_model(model, test_loaders, device, task_names=None):
    """Evaluate model on all tasks and compute continual learning metrics."""
    model.eval()
    
    results = {}
    accuracies = []
    
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            correct = 0
            total = 0
            
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            accuracy = 100.0 * correct / total
            accuracies.append(accuracy)
            
            task_name = task_names[task_id] if task_names else f"Task_{task_id}"
            results[task_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
    
    return results, accuracies


def compute_continual_learning_metrics(accuracy_matrix):
    """Compute standard continual learning metrics."""
    num_tasks = len(accuracy_matrix)
    
    final_accuracies = accuracy_matrix[-1]
    average_accuracy = np.mean(final_accuracies)
    
    backward_transfer = 0.0
    if num_tasks > 1:
        bwt_sum = 0.0
        for i in range(num_tasks - 1):
            bwt_sum += accuracy_matrix[-1][i] - accuracy_matrix[i][i]
        backward_transfer = bwt_sum / (num_tasks - 1)
    
    forward_transfer = 0.0
    if num_tasks > 1:
        fwt_sum = 0.0
        for i in range(1, num_tasks):
            random_baseline = 10.0 if len(accuracy_matrix[0]) == 10 else 20.0
            fwt_sum += accuracy_matrix[i-1][i] - random_baseline
        forward_transfer = fwt_sum / (num_tasks - 1)
    
    return {
        'average_accuracy': average_accuracy,
        'backward_transfer': backward_transfer,
        'forward_transfer': forward_transfer,
        'final_accuracies': final_accuracies
    }


def create_accuracy_heatmap(accuracy_matrix, method_name, save_path):
    """Create and save accuracy heatmap."""
    plt.figure(figsize=(10, 8))
    
    df = pd.DataFrame(accuracy_matrix, 
                     index=[f'After Task {i+1}' for i in range(len(accuracy_matrix))],
                     columns=[f'Task {i+1}' for i in range(len(accuracy_matrix[0]))])
    
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Accuracy (%)'})
    
    plt.title(f'Task Accuracy Matrix - {method_name}')
    plt.xlabel('Evaluated Task')
    plt.ylabel('Training Progress')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_comparison_plot(results_dict, save_path):
    """Create comparison plot of different methods."""
    methods = list(results_dict.keys())
    metrics = ['average_accuracy', 'backward_transfer', 'forward_transfer']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results_dict[method][metric] for method in methods]
        
        bars = axes[i].bar(methods, values, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Value')
        axes[i].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def create_learning_curves(training_losses, method_name, save_path):
    """Create and save learning curves."""
    plt.figure(figsize=(12, 6))
    
    for task_id, losses in enumerate(training_losses):
        plt.plot(losses, label=f'Task {task_id + 1}', alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Learning Curves - {method_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def create_forgetting_analysis(accuracy_matrix, save_path):
    """Create forgetting analysis plot."""
    num_tasks = len(accuracy_matrix)
    
    plt.figure(figsize=(12, 8))
    
    for task_id in range(num_tasks):
        task_accuracies = [accuracy_matrix[i][task_id] for i in range(task_id, num_tasks)]
        task_steps = list(range(task_id + 1, num_tasks + 1))
        
        plt.plot(task_steps, task_accuracies, 
                marker='o', label=f'Task {task_id + 1}', alpha=0.8)
    
    plt.xlabel('Training Step (After Task)')
    plt.ylabel('Accuracy (%)')
    plt.title('Forgetting Analysis: Task Performance Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_and_visualize(model, test_loaders, method_name, save_dir, 
                          training_losses=None, accuracy_matrix=None):
    """Complete evaluation and visualization pipeline."""
    device = next(model.parameters()).device
    
    results, accuracies = evaluate_model(model, test_loaders, device)
    
    if accuracy_matrix is not None:
        metrics = compute_continual_learning_metrics(accuracy_matrix)
        
        heatmap_path = f"{save_dir}/accuracy_heatmap_{method_name.lower()}.pdf"
        create_accuracy_heatmap(accuracy_matrix, method_name, heatmap_path)
        
        forgetting_path = f"{save_dir}/forgetting_analysis_{method_name.lower()}.pdf"
        create_forgetting_analysis(accuracy_matrix, forgetting_path)
        
        print(f"Results for {method_name}:")
        print(f"  Average Accuracy: {metrics['average_accuracy']:.2f}%")
        print(f"  Backward Transfer: {metrics['backward_transfer']:.2f}%")
        print(f"  Forward Transfer: {metrics['forward_transfer']:.2f}%")
        
        return metrics
    
    if training_losses is not None:
        curves_path = f"{save_dir}/learning_curves_{method_name.lower()}.pdf"
        create_learning_curves(training_losses, method_name, curves_path)
    
    return results
