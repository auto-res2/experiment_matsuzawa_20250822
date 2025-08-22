"""
CCAD-KD Evaluation Module
Implements comprehensive evaluation including worst-group analysis and calibration metrics.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocess import ContextManager, teacher_margins, ensure_dir


class CalibrationMetrics:
    """Expected Calibration Error and related metrics."""
    
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
    
    def compute_ece(self, confidences: np.ndarray, predictions: np.ndarray, 
                   targets: np.ndarray) -> Tuple[float, Dict]:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'prop_in_bin': prop_in_bin
                })
            else:
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'prop_in_bin': 0.0
                })
        
        return ece, {'bins': bin_data}


class WorstGroupAnalyzer:
    """Analyze worst-group performance across discovered contexts."""
    
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
    
    @torch.no_grad()
    def analyze_contexts(self, model: nn.Module, loader: DataLoader, 
                        device: str = 'cuda') -> Dict[str, Any]:
        """Analyze performance across contexts."""
        model.eval()
        
        ctx_correct = defaultdict(int)
        ctx_total = defaultdict(int)
        ctx_losses = defaultdict(list)
        ctx_confidences = defaultdict(list)
        
        all_predictions = []
        all_targets = []
        all_contexts = []
        
        for imgs, targets, aug_metas in tqdm(loader, desc="Analyzing contexts"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            ctx_ids = self.context_manager.get_context_ids(imgs, aug_metas)
            
            outputs = model(imgs)
            losses = F.cross_entropy(outputs, targets, reduction='none')
            probs = F.softmax(outputs, dim=1)
            confidences = probs.max(dim=1)[0]
            predictions = outputs.argmax(dim=1)
            
            for i, ctx in enumerate(ctx_ids):
                ctx_total[ctx] += 1
                ctx_correct[ctx] += (predictions[i] == targets[i]).item()
                ctx_losses[ctx].append(losses[i].item())
                ctx_confidences[ctx].append(confidences[i].item())
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_contexts.extend(ctx_ids)
        
        ctx_metrics = {}
        for ctx in ctx_total:
            if ctx_total[ctx] > 0:
                accuracy = ctx_correct[ctx] / ctx_total[ctx]
                avg_loss = np.mean(ctx_losses[ctx])
                avg_confidence = np.mean(ctx_confidences[ctx])
                
                ctx_metrics[ctx] = {
                    'accuracy': accuracy,
                    'loss': avg_loss,
                    'confidence': avg_confidence,
                    'count': ctx_total[ctx]
                }
        
        overall_accuracy = sum(ctx_correct.values()) / sum(ctx_total.values())
        
        accuracies = [m['accuracy'] for m in ctx_metrics.values()]
        worst_group_acc = min(accuracies) if accuracies else 0.0
        best_group_acc = max(accuracies) if accuracies else 0.0
        acc_std = np.std(accuracies) if len(accuracies) > 1 else 0.0
        
        return {
            'overall_accuracy': overall_accuracy,
            'worst_group_accuracy': worst_group_acc,
            'best_group_accuracy': best_group_acc,
            'accuracy_std': acc_std,
            'num_contexts': len(ctx_metrics),
            'context_metrics': ctx_metrics,
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'contexts': all_contexts
        }


class CCADEvaluator:
    """Comprehensive CCAD-KD evaluation."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.calibration = CalibrationMetrics()
        self.context_manager = ContextManager(device=device)
    
    def load_model(self, checkpoint_path: str, model: nn.Module) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['student_state_dict'])
        return model.to(self.device)
    
    @torch.no_grad()
    def evaluate_model(self, model: nn.Module, loader: DataLoader, 
                      fit_context_manager: bool = True) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        model.eval()
        
        if fit_context_manager and not self.context_manager.fitted:
            print("Fitting context manager for evaluation...")
            self.context_manager.fit_style_clusterer(loader)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_confidences = []
        all_predictions = []
        all_targets = []
        all_logits = []
        
        for imgs, targets, aug_metas in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, targets)
            
            probs = F.softmax(outputs, dim=1)
            confidences = probs.max(dim=1)[0]
            predictions = outputs.argmax(dim=1)
            
            total_loss += loss.item() * imgs.size(0)
            correct += predictions.eq(targets).sum().item()
            total += imgs.size(0)
            
            all_confidences.extend(confidences.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total
        
        confidences = np.array(all_confidences)
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        ece, calibration_data = self.calibration.compute_ece(confidences, predictions, targets)
        
        analyzer = WorstGroupAnalyzer(self.context_manager)
        context_analysis = analyzer.analyze_contexts(model, loader, self.device)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'ece': ece,
            'calibration_data': calibration_data,
            'context_analysis': context_analysis,
            'logits': np.vstack(all_logits) if all_logits else None
        }
    
    def compare_models(self, models: Dict[str, nn.Module], loader: DataLoader) -> Dict[str, Any]:
        """Compare multiple models."""
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            results[name] = self.evaluate_model(model, loader, fit_context_manager=(name == list(models.keys())[0]))
        
        return results
    
    def plot_calibration(self, results: Dict[str, Any], save_path: str):
        """Plot calibration curves."""
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
        if len(results) == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx]
            
            bins = result['calibration_data']['bins']
            accuracies = [b['accuracy'] for b in bins]
            confidences = [b['confidence'] for b in bins]
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
            ax.plot(confidences, accuracies, 'o-', label=f'{name} (ECE: {result["ece"]:.3f})')
            
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{name} Calibration')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Calibration plot saved to {save_path}")
    
    def plot_context_analysis(self, results: Dict[str, Any], save_path: str):
        """Plot context-wise performance analysis."""
        fig, axes = plt.subplots(2, len(results), figsize=(6*len(results), 8))
        if len(results) == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (name, result) in enumerate(results.items()):
            ctx_analysis = result['context_analysis']
            ctx_metrics = ctx_analysis['context_metrics']
            
            if not ctx_metrics:
                continue
            
            contexts = list(ctx_metrics.keys())
            accuracies = [ctx_metrics[c]['accuracy'] for c in contexts]
            counts = [ctx_metrics[c]['count'] for c in contexts]
            
            sorted_data = sorted(zip(contexts, accuracies, counts), key=lambda x: x[1])
            contexts, accuracies, counts = zip(*sorted_data)
            
            ax1 = axes[0, idx]
            bars = ax1.bar(range(len(contexts)), accuracies, alpha=0.7)
            ax1.axhline(y=ctx_analysis['overall_accuracy'], color='red', linestyle='--', 
                       label=f'Overall: {ctx_analysis["overall_accuracy"]:.3f}')
            ax1.axhline(y=ctx_analysis['worst_group_accuracy'], color='orange', linestyle='--',
                       label=f'Worst: {ctx_analysis["worst_group_accuracy"]:.3f}')
            ax1.set_xlabel('Context (sorted by accuracy)')
            ax1.set_ylabel('Accuracy')
            ax1.set_title(f'{name} - Context Accuracies')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[1, idx]
            ax2.bar(range(len(contexts)), counts, alpha=0.7, color='green')
            ax2.set_xlabel('Context')
            ax2.set_ylabel('Sample Count')
            ax2.set_title(f'{name} - Context Sample Counts')
            ax2.grid(True, alpha=0.3)
            
            if len(contexts) > 10:
                ax1.set_xticks([])
                ax2.set_xticks([])
        
        plt.tight_layout()
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Context analysis plot saved to {save_path}")
    
    def plot_training_curves(self, history: Dict[str, List[float]], save_path: str):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, history['val_accuracy'], label='Val Accuracy', color='green', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, history['beta'], label='Beta (intra-context weight)', alpha=0.8)
        axes[1, 0].plot(epochs, history['lambda'], label='Lambda (DRO weight)', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Parameter Value')
        axes[1, 0].set_title('CCAD Parameters')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, history['train_ce_loss'], label='CE Loss', alpha=0.8)
        axes[1, 1].plot(epochs, history['train_kd_loss'], label='KD Loss', alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Loss Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")
    
    def generate_report(self, results: Dict[str, Any], save_dir: str) -> str:
        """Generate evaluation report."""
        report_lines = []
        report_lines.append("CCAD-KD Evaluation Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        for name, result in results.items():
            report_lines.append(f"Model: {name}")
            report_lines.append("-" * 30)
            report_lines.append(f"Overall Accuracy: {result['accuracy']:.2f}%")
            report_lines.append(f"Average Loss: {result['loss']:.4f}")
            report_lines.append(f"Expected Calibration Error: {result['ece']:.4f}")
            
            ctx_analysis = result['context_analysis']
            report_lines.append(f"Worst-Group Accuracy: {ctx_analysis['worst_group_accuracy']:.4f}")
            report_lines.append(f"Best-Group Accuracy: {ctx_analysis['best_group_accuracy']:.4f}")
            report_lines.append(f"Accuracy Std Dev: {ctx_analysis['accuracy_std']:.4f}")
            report_lines.append(f"Number of Contexts: {ctx_analysis['num_contexts']}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        ensure_dir(save_dir)
        report_path = os.path.join(save_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"Evaluation report saved to {report_path}")
        return report_text


def evaluate_ccad_model(checkpoint_path: str, test_loader: DataLoader, 
                       model_architecture: str = 'resnet18', device: str = 'cuda',
                       save_dir: str = './.research/iteration1/images') -> Dict[str, Any]:
    """Evaluate a trained CCAD model."""
    try:
        import timm
        model = timm.create_model(model_architecture, pretrained=False, num_classes=100)
    except ImportError:
        import torchvision.models as models
        if model_architecture == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 100)
        else:
            raise ValueError(f"Unsupported architecture: {model_architecture}")
    
    evaluator = CCADEvaluator(device=device)
    model = evaluator.load_model(checkpoint_path, model)
    
    results = evaluator.evaluate_model(model, test_loader)
    
    evaluator.plot_calibration({'CCAD-KD': results}, 
                              os.path.join(save_dir, 'calibration_curve.pdf'))
    evaluator.plot_context_analysis({'CCAD-KD': results},
                                   os.path.join(save_dir, 'context_analysis.pdf'))
    
    evaluator.generate_report({'CCAD-KD': results}, save_dir)
    
    return results
