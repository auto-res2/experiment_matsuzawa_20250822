import torch
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from preprocess import create_continual_learning_datasets, get_data_loaders
from train import SimpleNet, SOCATrainer, BaselineTrainer
from evaluate import evaluate_and_visualize, compute_continual_learning_metrics, create_metrics_comparison_plot


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run_continual_learning_experiment(dataset_name, method_name, num_tasks=5, 
                                    epochs_per_task=3, device='cuda'):
    """Run a single continual learning experiment."""
    
    print(f"\n=== Running {method_name} on {dataset_name} ===")
    
    datasets = create_continual_learning_datasets(dataset_name, num_tasks)
    
    if dataset_name == 'permuted_mnist':
        input_size = 784
        num_classes = 10
    elif dataset_name == 'split_cifar100':
        input_size = 3 * 32 * 32
        num_classes = 100 // num_tasks  # classes per task
    
    model = SimpleNet(input_size, hidden_size=256, num_classes=num_classes).to(device)
    
    if method_name == 'SOCA':
        trainer = SOCATrainer(model, device, k_global=5, alpha=0.9, reg_lambda=0.001)
    elif method_name == 'Experience_Replay':
        trainer = BaselineTrainer(model, device, method='experience_replay')
    else:  # Finetune
        trainer = BaselineTrainer(model, device, method='finetune')
    
    accuracy_matrix = []
    training_losses = []
    test_loaders = []
    
    for task_id in range(num_tasks):
        print(f"\nTraining on Task {task_id + 1}/{num_tasks}")
        
        train_loader, test_loader = get_data_loaders(datasets[task_id], batch_size=32)
        test_loaders.append(test_loader)
        
        task_losses = []
        for epoch in range(epochs_per_task):
            loss = trainer.train_task(train_loader, epochs=1, lr=0.001)
            task_losses.append(loss)
            print(f"  Epoch {epoch + 1}: Loss = {loss:.4f}")
        
        training_losses.append(task_losses)
        
        model.eval()
        task_accuracies = []
        
        with torch.no_grad():
            for eval_task_id in range(task_id + 1):
                correct = 0
                total = 0
                
                for data, target in test_loaders[eval_task_id]:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                
                accuracy = 100.0 * correct / total
                task_accuracies.append(accuracy)
                print(f"  Task {eval_task_id + 1} Accuracy: {accuracy:.2f}%")
        
        while len(task_accuracies) < num_tasks:
            task_accuracies.append(0.0)
        
        accuracy_matrix.append(task_accuracies)
    
    return accuracy_matrix, training_losses, test_loaders, model


def main():
    """Main experimental pipeline."""
    print("Starting SOCA Continual Learning Experiments")
    print("=" * 50)
    
    set_random_seeds(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    output_dir = ".research/iteration1/images"
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ['permuted_mnist', 'split_cifar100']
    methods = ['SOCA', 'Experience_Replay', 'Finetune']
    num_tasks = 5  # Reduced for quick testing
    epochs_per_task = 2  # Reduced for quick testing
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        dataset_results = {}
        
        for method_name in methods:
            try:
                accuracy_matrix, training_losses, test_loaders, model = run_continual_learning_experiment(
                    dataset_name, method_name, num_tasks, epochs_per_task, device
                )
                
                metrics = compute_continual_learning_metrics(accuracy_matrix)
                dataset_results[method_name] = metrics
                
                save_dir = output_dir
                evaluate_and_visualize(
                    model, test_loaders, method_name, save_dir,
                    training_losses, accuracy_matrix
                )
                
                print(f"\n{method_name} Results:")
                print(f"  Average Accuracy: {metrics['average_accuracy']:.2f}%")
                print(f"  Backward Transfer: {metrics['backward_transfer']:.2f}%")
                print(f"  Forward Transfer: {metrics['forward_transfer']:.2f}%")
                
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error running {method_name} on {dataset_name}: {str(e)}")
                continue
        
        all_results[dataset_name] = dataset_results
        
        if dataset_results:
            comparison_path = f"{output_dir}/metrics_comparison_{dataset_name}.pdf"
            create_metrics_comparison_plot(dataset_results, comparison_path)
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        for method_name, metrics in dataset_results.items():
            print(f"  {method_name}:")
            print(f"    Avg Acc: {metrics['average_accuracy']:.2f}%")
            print(f"    BWT: {metrics['backward_transfer']:.2f}%")
            print(f"    FWT: {metrics['forward_transfer']:.2f}%")
    
    results_path = f"{output_dir}/experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nExperiment completed! Results saved to {output_dir}")
    print("Generated PDF plots:")
    
    pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
    for pdf_file in sorted(pdf_files):
        print(f"  - {pdf_file}")
    
    status_data = {"status_enum": "stopped", "timestamp": datetime.now().isoformat()}
    status_path = "experiment_status.json"
    with open(status_path, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    print(f"\nStatus set to 'stopped' in {status_path}")


if __name__ == "__main__":
    main()
