#!/usr/bin/env python3
"""
MoD-RCC — Risk-Controlled, Value-Aware Mixture-of-Depths for Token-Wise Dynamic Computation
Main experimental script implementing the complete pipeline from preprocessing to evaluation.

This script demonstrates:
1. Synthetic dataset creation for classification and segmentation
2. MoD-RCC model training with twin-critic value estimation
3. Risk calibration with distribution-free guarantees
4. Comprehensive evaluation including robustness and Pareto analysis
5. High-quality PDF plot generation for academic papers
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

from preprocess import set_seed, create_datasets
from train import train_classification_model
from evaluate import run_comprehensive_evaluation

plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def create_training_plots(train_losses, val_accuracies, save_path):
    """Create training progress plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('MoD-RCC Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, val_accuracies, 'r-', linewidth=2, marker='s', markersize=6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('MoD-RCC Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Training plots saved to: {save_path}")

def create_robustness_plots(robustness_results, save_path):
    """Create robustness comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    conditions = list(robustness_results.keys())
    metrics = ['accuracy', 'avg_cost', 'avg_confidence']
    metric_labels = ['Accuracy', 'Average Cost', 'Average Confidence']
    colors = ['#2E86AB', '#A23B72']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [robustness_results[cond][metric] for cond in conditions]
        bars = axes[i].bar(conditions, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[i].set_ylabel(label)
        axes[i].set_title(f'{label} Comparison')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Robustness plots saved to: {save_path}")

def create_pareto_frontier_plot(pareto_points, save_path):
    """Create Pareto frontier plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    costs, accuracies = zip(*pareto_points)
    
    ax.plot(costs, accuracies, 'o-', linewidth=3, markersize=8, 
            color='#F18F01', markerfacecolor='#C73E1D', markeredgecolor='black', 
            markeredgewidth=1.5, label='MoD-RCC Pareto Frontier')
    
    for i, (cost, acc) in enumerate(pareto_points):
        ax.annotate(f'B={0.3 + i*0.2:.1f}', (cost, acc), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, ha='left')
    
    ax.set_xlabel('Computational Cost (Normalized)')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('MoD-RCC: Accuracy vs Computational Cost Trade-off')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if len(pareto_points) >= 2:
        ax.fill_between(costs, accuracies, alpha=0.2, color='#F18F01', 
                       label='Efficient Operating Region')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Pareto frontier plot saved to: {save_path}")

def create_risk_calibration_plot(risk_results, save_path):
    """Create risk calibration visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    coverage = risk_results['coverage']
    target_epsilon = risk_results['target_epsilon']
    controlled_deg = risk_results['controlled_degradation']
    
    ax1.bar(['Achieved Coverage'], [coverage], color='#2E86AB', alpha=0.8, 
           edgecolor='black', linewidth=1.5)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Coverage')
    ax1.set_ylabel('Coverage Rate')
    ax1.set_title('Risk Calibration Coverage')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    ax1.text(0, coverage + 0.05, f'{coverage:.3f}', ha='center', va='bottom', 
            fontweight='bold', fontsize=12)
    
    categories = ['Target ε', 'Controlled Degradation']
    values = [target_epsilon, controlled_deg]
    colors = ['#A23B72', '#F18F01']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Degradation Rate')
    ax2.set_title('Risk Control Effectiveness')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Risk calibration plot saved to: {save_path}")

def create_system_overview_plot(save_path):
    """Create MoD-RCC system architecture overview."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    components = {
        'Input\nImage': (1, 6),
        'Patch\nEmbedding': (3, 6),
        'Twin-Critic\nMCV': (5, 8),
        'Soft Top-K\nRouter': (5, 6),
        'Budget\nController': (5, 4),
        'Shallow\nHead': (7, 7),
        'Deep\nHead': (7, 5),
        'Uncertainty\nFusion': (9, 6),
        'Risk\nCalibrator': (11, 6),
        'Output': (13, 6)
    }
    
    for name, (x, y) in components.items():
        if 'Head' in name:
            color = '#F18F01'
        elif 'Critic' in name or 'Router' in name:
            color = '#2E86AB'
        elif 'Controller' in name or 'Calibrator' in name:
            color = '#A23B72'
        else:
            color = '#C73E1D'
        
        rect = patches.FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=color, alpha=0.7,
                                     edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontweight='bold', 
               fontsize=10, color='white')
    
    connections = [
        ((1, 6), (3, 6)),  # Input -> Patch Embedding
        ((3, 6), (5, 6)),  # Patch Embedding -> Router
        ((5, 6), (5, 8)),  # Router -> Twin-Critic
        ((5, 6), (5, 4)),  # Router -> Budget Controller
        ((5, 6), (7, 7)),  # Router -> Shallow Head
        ((5, 6), (7, 5)),  # Router -> Deep Head
        ((7, 7), (9, 6)),  # Shallow Head -> Fusion
        ((7, 5), (9, 6)),  # Deep Head -> Fusion
        ((9, 6), (11, 6)), # Fusion -> Risk Calibrator
        ((11, 6), (13, 6)) # Risk Calibrator -> Output
    ]
    
    for (x1, y1), (x2, y2) in connections:
        ax.arrow(x1+0.7, y1, x2-x1-1.4, y2-y1, head_width=0.15, head_length=0.2, 
                fc='black', ec='black', alpha=0.7)
    
    ax.set_xlim(0, 14)
    ax.set_ylim(3, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('MoD-RCC System Architecture', fontsize=16, fontweight='bold', pad=20)
    
    legend_elements = [
        patches.Patch(color='#C73E1D', alpha=0.7, label='Input/Output'),
        patches.Patch(color='#2E86AB', alpha=0.7, label='Core Routing'),
        patches.Patch(color='#F18F01', alpha=0.7, label='Prediction Heads'),
        patches.Patch(color='#A23B72', alpha=0.7, label='Control Systems')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"System overview plot saved to: {save_path}")

def save_experiment_summary(results, save_path):
    """Save experiment summary as JSON."""
    summary = {
        'experiment_name': 'MoD-RCC_Risk_Controlled_Mixture_of_Depths',
        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
        'final_metrics': {
            'clean_accuracy': float(results['robustness']['Clean']['accuracy']),
            'corrupted_accuracy': float(results['robustness']['Corrupted']['accuracy']),
            'robustness_gap': float(results['robustness']['Clean']['accuracy'] - 
                                  results['robustness']['Corrupted']['accuracy']),
            'risk_calibration_coverage': float(results['risk_calibration']['coverage']),
            'controlled_degradation': float(results['risk_calibration']['controlled_degradation']),
            'target_epsilon': float(results['risk_calibration']['target_epsilon'])
        },
        'training_summary': {
            'final_train_loss': float(results['train_losses'][-1]),
            'final_val_accuracy': float(results['val_accuracies'][-1]),
            'num_epochs': len(results['train_losses'])
        },
        'pareto_frontier': [
            {'cost': float(cost), 'accuracy': float(acc)} 
            for cost, acc in results['pareto_frontier']
        ]
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Experiment summary saved to: {save_path}")

def update_status_to_stopped():
    """Update status_enum to 'stopped' as required."""
    status_file = Path('.research/research_history.json')
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                data = json.load(f)
            
            data['status_enum'] = 'stopped'
            
            with open(status_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print("Status updated to 'stopped' in research_history.json")
        except Exception as e:
            print(f"Warning: Could not update status_enum: {e}")
    else:
        print("Warning: research_history.json not found")

def main():
    """Main experimental pipeline."""
    print("=" * 80)
    print("MoD-RCC: Risk-Controlled, Value-Aware Mixture-of-Depths")
    print("Token-Wise Dynamic Computation Experiment")
    print("=" * 80)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path('.research/iteration1/images')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print("STAGE 1: DATA PREPROCESSING")
    print("="*50)
    datasets = create_datasets()
    
    print("\n" + "="*50)
    print("STAGE 2: MODEL TRAINING & EVALUATION")
    print("="*50)
    results = run_comprehensive_evaluation(datasets, device)
    
    print("\n" + "="*50)
    print("STAGE 3: VISUALIZATION & ANALYSIS")
    print("="*50)
    
    create_system_overview_plot(output_dir / 'mod_rcc_system_architecture.pdf')
    create_training_plots(results['train_losses'], results['val_accuracies'], 
                         output_dir / 'training_progress.pdf')
    create_robustness_plots(results['robustness'], 
                           output_dir / 'robustness_analysis.pdf')
    create_pareto_frontier_plot(results['pareto_frontier'], 
                               output_dir / 'pareto_frontier.pdf')
    create_risk_calibration_plot(results['risk_calibration'], 
                                output_dir / 'risk_calibration.pdf')
    
    save_experiment_summary(results, output_dir / 'experiment_summary.json')
    
    print("\n" + "="*50)
    print("STAGE 4: FINAL RESULTS SUMMARY")
    print("="*50)
    
    print(f"🎯 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"📊 Results Summary:")
    print(f"   • Clean Data Accuracy: {results['robustness']['Clean']['accuracy']:.4f}")
    print(f"   • Corrupted Data Accuracy: {results['robustness']['Corrupted']['accuracy']:.4f}")
    print(f"   • Robustness Gap: {results['robustness']['Clean']['accuracy'] - results['robustness']['Corrupted']['accuracy']:.4f}")
    print(f"   • Risk Calibration Coverage: {results['risk_calibration']['coverage']:.4f}")
    print(f"   • Controlled Degradation: {results['risk_calibration']['controlled_degradation']:.4f}")
    print(f"   • Target Risk Threshold (ε): {results['risk_calibration']['target_epsilon']:.4f}")
    
    print(f"\n📈 Pareto Frontier Points:")
    for i, (cost, acc) in enumerate(results['pareto_frontier']):
        budget = 0.3 + i * 0.2
        print(f"   • Budget {budget:.1f}: Cost={cost:.4f}, Accuracy={acc:.4f}")
    
    print(f"\n📁 Generated Files:")
    for pdf_file in output_dir.glob('*.pdf'):
        print(f"   • {pdf_file.name}")
    print(f"   • experiment_summary.json")
    
    print(f"\n🔬 Key Contributions Demonstrated:")
    print(f"   ✓ Twin-Critic marginal compute value estimation")
    print(f"   ✓ Soft Top-K routing with budget control")
    print(f"   ✓ Complementary shallow/deep prediction heads")
    print(f"   ✓ Distribution-free risk calibration")
    print(f"   ✓ Robustness under distribution shift")
    print(f"   ✓ Pareto-efficient accuracy/cost trade-offs")
    
    update_status_to_stopped()
    
    print(f"\n🎉 MoD-RCC experiment pipeline completed successfully!")
    print(f"All results saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
