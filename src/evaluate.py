"""
AccuTune Evaluation Module
Handles model evaluation, metrics computation, and visualization generation.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def evaluate_model(model, test_loader, device="cuda", criterion=None, quick_test=False):
    """Evaluate model performance on test set"""
    print(f"Evaluating model on {device}")
    
    model.eval()
    model = model.to(device)
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            if quick_test and batch_idx >= 20:  # Early stop for quick test
                break
    
    eval_time = time.time() - start_time
    
    avg_loss = total_loss / max(len(test_loader), 1)
    accuracy = 100 * correct / max(total, 1)
    
    results = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "eval_time": eval_time,
        "predictions": all_predictions,
        "targets": all_targets
    }
    
    print(f"Evaluation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Time: {eval_time:.2f}s")
    
    return results


def compute_detailed_metrics(predictions, targets, num_classes=10):
    """Compute detailed classification metrics"""
    if not HAS_SKLEARN:
        print("scikit-learn not available, skipping detailed metrics")
        return {}
    
    try:
        cm = confusion_matrix(targets, predictions, labels=range(num_classes))
        
        report = classification_report(targets, predictions, 
                                     labels=range(num_classes), 
                                     output_dict=True, 
                                     zero_division=0)
        
        per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
        
        return {
            "confusion_matrix": cm,
            "classification_report": report,
            "per_class_accuracy": per_class_acc,
            "macro_avg_f1": report.get("macro avg", {}).get("f1-score", 0),
            "weighted_avg_f1": report.get("weighted avg", {}).get("f1-score", 0)
        }
    except Exception as e:
        print(f"Error computing detailed metrics: {e}")
        return {}


def analyze_accumulator_telemetry(model):
    """Analyze telemetry from low-bit accumulator layers"""
    from train import LowBitAccLinear
    
    telemetry = {}
    layer_idx = 0
    
    for name, module in model.named_modules():
        if isinstance(module, LowBitAccLinear):
            layer_data = {
                "name": name,
                "mode": module.mode,
                "order": module.order,
                "overflow_count": int(module.of_count.item()),
                "underflow_count": int(module.uf_count.item()),
                "swamping_count": int(module.swamp_count.item()),
                "headroom": float(module.headroom.item()),
                "alpha_ema": float(module.alpha_ema.item()),
                "mantissa_hist": module.mantissa_hist.cpu().numpy()
            }
            telemetry[f"layer_{layer_idx}"] = layer_data
            layer_idx += 1
    
    return telemetry


def create_training_plots(results, save_dir="./plots", experiment_name="experiment"):
    """Create and save training visualization plots as PDFs"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(results["train_losses"]) + 1)
    
    ax1.plot(epochs, results["train_losses"], 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs, results["test_accuracies"], 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{experiment_name}_training_curves.pdf", 
                format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    if "ga_history" in results and results["ga_history"]:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(epochs, results["ga_history"], 'g-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Accumulation Steps')
        ax.set_title('Dynamic Gradient Accumulation')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{experiment_name}_ga_history.pdf", 
                    format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    if "energy_history" in results and results["energy_history"]:
        energy_data = results["energy_history"]
        if energy_data:
            times = range(len(energy_data))
            powers = [e["power_avg"] for e in energy_data]
            energies = [e["energy_total"] for e in energy_data]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(times, powers, 'orange', linewidth=2)
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Power (W)')
            ax1.set_title('Average Power Consumption')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(times, energies, 'purple', linewidth=2)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Total Energy (J)')
            ax2.set_title('Cumulative Energy Consumption')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{experiment_name}_energy.pdf", 
                        format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Training plots saved to {save_dir}/")


def create_evaluation_plots(eval_results, detailed_metrics, save_dir="./plots", experiment_name="experiment"):
    """Create and save evaluation visualization plots as PDFs"""
    os.makedirs(save_dir, exist_ok=True)
    
    if "confusion_matrix" in detailed_metrics:
        cm = detailed_metrics["confusion_matrix"]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{experiment_name}_confusion_matrix.pdf", 
                    format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    if "per_class_accuracy" in detailed_metrics:
        per_class_acc = detailed_metrics["per_class_accuracy"]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        classes = range(len(per_class_acc))
        ax.bar(classes, per_class_acc * 100, alpha=0.7, color='skyblue')
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Per-Class Accuracy')
        ax.set_xticks(classes)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{experiment_name}_per_class_accuracy.pdf", 
                    format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Evaluation plots saved to {save_dir}/")


def create_telemetry_plots(telemetry, save_dir="./plots", experiment_name="experiment"):
    """Create and save accumulator telemetry plots as PDFs"""
    if not telemetry:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    layer_names = list(telemetry.keys())
    of_counts = [telemetry[l]["overflow_count"] for l in layer_names]
    uf_counts = [telemetry[l]["underflow_count"] for l in layer_names]
    swamp_counts = [telemetry[l]["swamping_count"] for l in layer_names]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    x = np.arange(len(layer_names))
    width = 0.25
    
    ax.bar(x - width, of_counts, width, label='Overflow', alpha=0.8)
    ax.bar(x, uf_counts, width, label='Underflow', alpha=0.8)
    ax.bar(x + width, swamp_counts, width, label='Swamping', alpha=0.8)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Count')
    ax.set_title('Accumulator Error Counts by Layer')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{experiment_name}_telemetry_counts.pdf", 
                format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    alpha_values = [telemetry[l]["alpha_ema"] for l in layer_names]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(range(len(layer_names)), alpha_values, alpha=0.7, color='green')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Alpha EMA')
    ax.set_title('DIFF-lite Correction Factors by Layer')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{experiment_name}_alpha_ema.pdf", 
                format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Telemetry plots saved to {save_dir}/")


def run_full_evaluation(model, test_loader, results, save_dir="./plots", 
                       experiment_name="experiment", device="cuda", quick_test=False):
    """Run complete evaluation pipeline with all visualizations"""
    print(f"Running full evaluation for {experiment_name}")
    
    eval_results = evaluate_model(model, test_loader, device=device, quick_test=quick_test)
    
    detailed_metrics = compute_detailed_metrics(
        eval_results["predictions"], 
        eval_results["targets"]
    )
    
    telemetry = analyze_accumulator_telemetry(model)
    
    create_training_plots(results, save_dir, experiment_name)
    create_evaluation_plots(eval_results, detailed_metrics, save_dir, experiment_name)
    create_telemetry_plots(telemetry, save_dir, experiment_name)
    
    summary = {
        "experiment_name": experiment_name,
        "final_accuracy": eval_results["accuracy"],
        "final_loss": eval_results["loss"],
        "eval_time": eval_results["eval_time"],
        "num_lowbit_layers": len(telemetry),
        "total_overflow_events": sum(t["overflow_count"] for t in telemetry.values()),
        "total_underflow_events": sum(t["underflow_count"] for t in telemetry.values()),
        "total_swamping_events": sum(t["swamping_count"] for t in telemetry.values()),
    }
    
    if detailed_metrics:
        summary.update({
            "macro_f1": detailed_metrics.get("macro_avg_f1", 0),
            "weighted_f1": detailed_metrics.get("weighted_avg_f1", 0)
        })
    
    print(f"\nEvaluation Summary for {experiment_name}:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return summary


if __name__ == "__main__":
    print("Testing AccuTune evaluation components...")
    
    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    )
    
    test_data = torch.randn(100, 32)
    test_targets = torch.randint(0, 10, (100,))
    test_dataset = torch.utils.data.TensorDataset(test_data, test_targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
    
    dummy_results = {
        "train_losses": [2.3, 1.8, 1.5, 1.2, 1.0],
        "test_accuracies": [20, 35, 50, 65, 75],
        "ga_history": [1, 2, 2, 4, 4],
        "energy_history": [
            {"power_avg": 150, "energy_total": 100},
            {"power_avg": 160, "energy_total": 250},
            {"power_avg": 155, "energy_total": 400}
        ]
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = run_full_evaluation(
        model, test_loader, dummy_results, 
        save_dir="./test_plots", 
        experiment_name="test", 
        device=device,
        quick_test=True
    )
    
    print("Evaluation components test completed successfully!")
