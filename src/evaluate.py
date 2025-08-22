#!/usr/bin/env python3
"""
Evaluation utilities for Rev-SS2D experiments
Includes model evaluation, gradient comparison, and plotting functions
"""

import os
import math
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """Evaluation utilities for Rev-SS2D experiments"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        plt.style.use('default')
        sns.set_palette("husl")
        
    def compare_gradients(self, model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
        """Compare gradients between two models for correctness validation"""
        
        grad_diffs = []
        param_names = []
        
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        
        common_params = set(params1.keys()) & set(params2.keys())
        
        for name in common_params:
            p1, p2 = params1[name], params2[name]
            
            if p1.grad is not None and p2.grad is not None:
                diff = torch.abs(p1.grad - p2.grad)
                norm1 = torch.norm(p1.grad)
                norm2 = torch.norm(p2.grad)
                
                if norm1 > 1e-8 and norm2 > 1e-8:
                    rel_error = torch.max(diff) / torch.max(norm1, norm2)
                    grad_diffs.append(rel_error.item())
                    param_names.append(name)
        
        if grad_diffs:
            max_rel_error = max(grad_diffs)
            mean_rel_error = sum(grad_diffs) / len(grad_diffs)
        else:
            max_rel_error = float('nan')
            mean_rel_error = float('nan')
        
        return {
            'max_rel_error': max_rel_error,
            'mean_rel_error': mean_rel_error,
            'num_compared_params': len(grad_diffs),
            'param_names': param_names
        }
    
    def evaluate_model_accuracy(
        self, 
        model: nn.Module, 
        data_loader: torch.utils.data.DataLoader,
        num_classes: int
    ) -> Dict[str, Any]:
        """Evaluate model accuracy on a dataset"""
        
        model.eval()
        correct = 0
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_correct[label] += (predicted[i] == labels[i]).item()
                    class_total[label] += 1
        
        overall_accuracy = 100 * correct / total if total > 0 else 0
        
        class_accuracies = []
        for i in range(num_classes):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        return {
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies,
            'total_samples': total
        }
    
    def measure_inference_speed(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, int, int, int],
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """Measure model inference speed"""
        
        model.eval()
        dummy_input = torch.randn(input_shape, device=self.device)
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time.record()
                    _ = model(dummy_input)
                    end_time.record()
                    torch.cuda.synchronize()
                    times.append(start_time.elapsed_time(end_time))
        else:
            import time
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = model(dummy_input)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        batch_size = input_shape[0]
        throughput = batch_size * 1000 / mean_time  # samples/sec
        
        return {
            'mean_time_ms': mean_time,
            'std_time_ms': std_time,
            'throughput_samples_per_sec': throughput
        }
    
    def plot_memory_comparison(self, results: List[Dict], save_path: str):
        """Plot memory usage comparison between baseline and Rev-SS2D"""
        
        baseline_data = [r for r in results if r['name'] == 'Baseline']
        revss2d_data = [r for r in results if r['name'] == 'Rev-SS2D']
        
        resolutions = sorted(list(set([r['resolution'] for r in results])))
        
        baseline_memory = []
        revss2d_memory = []
        
        for res in resolutions:
            baseline_mem = next((r['peak_memory_gb'] for r in baseline_data if r['resolution'] == res), float('inf'))
            revss2d_mem = next((r['peak_memory_gb'] for r in revss2d_data if r['resolution'] == res), float('inf'))
            
            baseline_memory.append(baseline_mem)
            revss2d_memory.append(revss2d_mem)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.arange(len(resolutions))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_memory, width, label='Baseline', alpha=0.8, color='#ff7f0e')
        bars2 = ax1.bar(x + width/2, revss2d_memory, width, label='Rev-SS2D', alpha=0.8, color='#2ca02c')
        
        ax1.set_xlabel('Resolution')
        ax1.set_ylabel('Peak Memory (GB)')
        ax1.set_title('Memory Usage Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{r}x{r}' for r in resolutions])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            if not math.isinf(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
            else:
                ax1.text(bar.get_x() + bar.get_width()/2., 0.5,
                        'OOM', ha='center', va='bottom', fontsize=9, color='red')
        
        for bar in bars2:
            height = bar.get_height()
            if not math.isinf(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
            else:
                ax1.text(bar.get_x() + bar.get_width()/2., 0.5,
                        'OOM', ha='center', va='bottom', fontsize=9, color='red')
        
        reductions = []
        valid_resolutions = []
        
        for i, res in enumerate(resolutions):
            if not math.isinf(baseline_memory[i]) and not math.isinf(revss2d_memory[i]) and revss2d_memory[i] > 0:
                reduction = baseline_memory[i] / revss2d_memory[i]
                reductions.append(reduction)
                valid_resolutions.append(res)
        
        if reductions:
            bars3 = ax2.bar(range(len(valid_resolutions)), reductions, alpha=0.8, color='#1f77b4')
            ax2.set_xlabel('Resolution')
            ax2.set_ylabel('Memory Reduction Factor')
            ax2.set_title('Memory Reduction (Baseline / Rev-SS2D)')
            ax2.set_xticks(range(len(valid_resolutions)))
            ax2.set_xticklabels([f'{r}x{r}' for r in valid_resolutions])
            ax2.grid(True, alpha=0.3)
            
            for i, bar in enumerate(bars3):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No valid comparisons\n(OOM errors)', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Memory Reduction (Baseline / Rev-SS2D)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Memory comparison plot saved to {save_path}")
    
    def plot_ablation_study(self, results: List[Dict], save_path: str):
        """Plot ablation study results"""
        
        configs = [r['config'] for r in results]
        memory_usage = [r['peak_memory_gb'] for r in results]
        final_losses = [r['final_loss'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(configs)))
        bars1 = ax1.bar(range(len(configs)), memory_usage, color=colors, alpha=0.8)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Peak Memory (GB)')
        ax1.set_title('Memory Usage by Configuration')
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            if not math.isinf(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            else:
                ax1.text(bar.get_x() + bar.get_width()/2., 0.5,
                        'OOM', ha='center', va='bottom', fontsize=9, color='red')
        
        bars2 = ax2.bar(range(len(configs)), final_losses, color=colors, alpha=0.8)
        
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Final Loss')
        ax2.set_title('Training Loss by Configuration')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            if not math.isnan(height):
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Ablation study plot saved to {save_path}")
    
    def plot_training_curves(self, results: List[Dict], save_path: str):
        """Plot training loss curves for different configurations"""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(results)))
        
        for i, result in enumerate(results):
            config = result['config']
            losses = result['losses']
            
            if not any(math.isnan(loss) for loss in losses):
                steps = list(range(1, len(losses) + 1))
                ax.plot(steps, losses, marker='o', label=config, color=colors[i], linewidth=2, markersize=4)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curves by Configuration')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Training curves plot saved to {save_path}")
    
    def plot_accuracy_curves(self, train_accs: List[float], val_accs: List[float], save_path: str):
        """Plot training and validation accuracy curves"""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = list(range(1, len(train_accs) + 1))
        
        ax.plot(epochs, train_accs, marker='o', label='Training Accuracy', linewidth=2, markersize=4)
        ax.plot(epochs, val_accs, marker='s', label='Validation Accuracy', linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training and Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        best_train = max(train_accs)
        best_val = max(val_accs)
        best_train_epoch = train_accs.index(best_train) + 1
        best_val_epoch = val_accs.index(best_val) + 1
        
        ax.annotate(f'Best Train: {best_train:.1f}%', 
                   xy=(best_train_epoch, best_train), 
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate(f'Best Val: {best_val:.1f}%', 
                   xy=(best_val_epoch, best_val), 
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Accuracy curves plot saved to {save_path}")
    
    def create_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        class_names: List[str],
        save_path: str
    ):
        """Create and save confusion matrix plot"""
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
        if ax.figure is not None:
            ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def generate_experiment_report(self, results: Dict[str, Any], save_path: str):
        """Generate a comprehensive experiment report"""
        
        report_lines = [
            "# Rev-SS2D Experiment Report",
            "=" * 50,
            "",
            f"**Experiment Timestamp:** {results.get('timestamp', 'N/A')}",
            f"**Device:** {results.get('device', 'N/A')}",
            "",
            "## Memory Benchmark Results",
            "-" * 30,
        ]
        
        if 'memory_benchmark' in results:
            memory_results = results['memory_benchmark']
            
            baseline_results = [r for r in memory_results if r['name'] == 'Baseline']
            revss2d_results = [r for r in memory_results if r['name'] == 'Rev-SS2D']
            
            for baseline, revss2d in zip(baseline_results, revss2d_results):
                if baseline['resolution'] == revss2d['resolution']:
                    res = baseline['resolution']
                    baseline_mem = baseline['peak_memory_gb']
                    revss2d_mem = revss2d['peak_memory_gb']
                    
                    if not math.isinf(baseline_mem) and not math.isinf(revss2d_mem) and revss2d_mem > 0:
                        reduction = baseline_mem / revss2d_mem
                        report_lines.extend([
                            f"**Resolution {res}x{res}:**",
                            f"  - Baseline Memory: {baseline_mem:.2f} GB",
                            f"  - Rev-SS2D Memory: {revss2d_mem:.2f} GB",
                            f"  - Memory Reduction: {reduction:.2f}x",
                            ""
                        ])
                    else:
                        report_lines.extend([
                            f"**Resolution {res}x{res}:**",
                            f"  - Baseline Memory: {'OOM' if math.isinf(baseline_mem) else f'{baseline_mem:.2f} GB'}",
                            f"  - Rev-SS2D Memory: {'OOM' if math.isinf(revss2d_mem) else f'{revss2d_mem:.2f} GB'}",
                            ""
                        ])
        
        if 'gradient_correctness' in results:
            grad_results = results['gradient_correctness']
            report_lines.extend([
                "## Gradient Correctness Test",
                "-" * 30,
                f"**Test Status:** {'PASSED' if grad_results.get('correctness_passed', False) else 'FAILED'}",
                f"**Max Gradient Relative Error:** {grad_results.get('max_grad_error', 'N/A'):.6f}",
                f"**Mean Gradient Relative Error:** {grad_results.get('mean_grad_error', 'N/A'):.6f}",
                f"**Baseline Loss:** {grad_results.get('baseline_loss', 'N/A'):.6f}",
                f"**Rev-SS2D Loss:** {grad_results.get('rev_loss', 'N/A'):.6f}",
                ""
            ])
        
        if 'ablation_study' in results:
            ablation_results = results['ablation_study']
            report_lines.extend([
                "## Ablation Study Results",
                "-" * 30,
            ])
            
            for result in ablation_results:
                config = result['config']
                memory = result['peak_memory_gb']
                loss = result['final_loss']
                
                report_lines.extend([
                    f"**{config}:**",
                    f"  - Peak Memory: {'OOM' if math.isinf(memory) else f'{memory:.2f} GB'}",
                    f"  - Final Loss: {loss:.4f}" if not math.isnan(loss) else "  - Final Loss: Failed",
                    ""
                ])
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Experiment report saved to {save_path}")


class MetricsTracker:
    """Track and log training metrics"""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'memory_usage': [],
            'step_times': []
        }
    
    def update(self, **kwargs):
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_latest(self, metric_name: str):
        """Get the latest value of a metric"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
    
    def get_best(self, metric_name: str, mode: str = 'max'):
        """Get the best value of a metric"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            if mode == 'max':
                return max(self.metrics[metric_name])
            else:
                return min(self.metrics[metric_name])
        return None
    
    def save_metrics(self, save_path: str):
        """Save metrics to file"""
        import json
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {save_path}")
