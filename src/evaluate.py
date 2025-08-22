import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple
import os

from train import TinySSMModel, to_device, save_pdf


def evaluate_model(model: TinySSMModel, test_loader, device: torch.device) -> Dict:
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch, device)
            assert isinstance(batch, dict)
            logits = model(batch['tokens'])
            loss = F.cross_entropy(logits, batch['labels'])
            test_loss += loss.item()
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_loss /= len(test_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def analyze_spectral_properties(model: TinySSMModel, device: torch.device, L0: int = 512) -> Dict:
    model.eval()
    model.to(device)
    
    spectral_analysis = {}
    
    with torch.no_grad():
        for i, layer in enumerate(model.get_ssm_layers()):
            kernel = layer.get_kernel(L0)
            
            kernel_fft = torch.fft.rfft(kernel)
            psd = (kernel_fft.abs() ** 2).cpu().numpy()
            
            freqs = np.fft.rfftfreq(L0)
            
            decay_params = torch.exp(layer.log_decay).cpu().numpy()
            omega_params = layer.omega.cpu().numpy()
            alpha_params = F.softplus(layer.log_alpha).cpu().numpy()
            
            spectral_analysis[f'layer_{i}'] = {
                'psd': psd,
                'frequencies': freqs,
                'decay_params': decay_params,
                'omega_params': omega_params,
                'alpha_params': alpha_params,
                'kernel': kernel.cpu().numpy()
            }
    
    return spectral_analysis


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        val_accuracies: List[float], save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_pdf(fig, save_path)


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    save_pdf(fig, save_path)


def plot_spectral_analysis(spectral_data: Dict, save_path: str):
    n_layers = len([k for k in spectral_data.keys() if k.startswith('layer_')])
    
    fig, axes = plt.subplots(2, n_layers, figsize=(4*n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_layers):
        layer_data = spectral_data[f'layer_{i}']
        
        axes[0, i].semilogy(layer_data['frequencies'], layer_data['psd'])
        axes[0, i].set_xlabel('Frequency')
        axes[0, i].set_ylabel('Power Spectral Density')
        axes[0, i].set_title(f'Layer {i} PSD')
        axes[0, i].grid(True, alpha=0.3)
        
        axes[1, i].plot(layer_data['kernel'][:100])
        axes[1, i].set_xlabel('Time')
        axes[1, i].set_ylabel('Kernel Value')
        axes[1, i].set_title(f'Layer {i} Kernel (first 100 steps)')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_pdf(fig, save_path)


def plot_parameter_analysis(spectral_data: Dict, save_path: str):
    n_layers = len([k for k in spectral_data.keys() if k.startswith('layer_')])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    decay_params = []
    omega_params = []
    alpha_params = []
    
    for i in range(n_layers):
        layer_data = spectral_data[f'layer_{i}']
        decay_params.extend(layer_data['decay_params'])
        omega_params.extend(layer_data['omega_params'])
        alpha_params.extend(layer_data['alpha_params'])
    
    axes[0].hist(decay_params, bins=20, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Decay Parameter')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Decay Parameters')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(omega_params, bins=20, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Omega Parameter')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Omega Parameters')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(alpha_params, bins=20, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Alpha Parameter')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Distribution of Alpha Parameters')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_pdf(fig, save_path)


def comprehensive_evaluation(model: TinySSMModel, test_loader, device: torch.device, 
                           train_losses: List[float], val_losses: List[float], 
                           val_accuracies: List[float], output_dir: str) -> Dict:
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Evaluating model performance...")
    eval_results = evaluate_model(model, test_loader, device)
    
    print("Analyzing spectral properties...")
    spectral_data = analyze_spectral_properties(model, device)
    
    print("Generating plots...")
    plot_training_curves(train_losses, val_losses, val_accuracies, 
                         os.path.join(output_dir, 'training_curves.pdf'))
    
    plot_confusion_matrix(eval_results['labels'], eval_results['predictions'],
                         os.path.join(output_dir, 'confusion_matrix.pdf'))
    
    plot_spectral_analysis(spectral_data, 
                          os.path.join(output_dir, 'spectral_analysis.pdf'))
    
    plot_parameter_analysis(spectral_data,
                           os.path.join(output_dir, 'parameter_analysis.pdf'))
    
    print(f"Test Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Test Loss: {eval_results['test_loss']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(eval_results['labels'], eval_results['predictions']))
    
    return {
        'evaluation': eval_results,
        'spectral_analysis': spectral_data,
        'plots_saved': True
    }
