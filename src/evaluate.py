import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

def power_iteration_sigma_max(layer, x, num_iterations=20):
    """Estimate maximum singular value of layer's Jacobian using power iteration."""
    if not isinstance(x, torch.Tensor):
        return 0.0
    
    if x.dim() > 2:
        x = x[0:1]
    
    v = torch.randn_like(x)
    v = v / torch.norm(v)
    
    layer.eval()
    
    try:
        with torch.enable_grad():
            x_detached = x.detach().requires_grad_(True)
            
            for _ in range(num_iterations):
                layer_output = layer(x_detached)
                
                jvp = torch.autograd.grad(
                    layer_output, x_detached, 
                    grad_outputs=v, 
                    retain_graph=True, 
                    create_graph=True
                )[0]
                
                v = torch.autograd.grad(
                    layer_output, x_detached, 
                    grad_outputs=jvp, 
                    retain_graph=False
                )[0]
                
                v_norm = torch.norm(v)
                if v_norm > 1e-8:
                    v = v / v_norm
                else:
                    break
            
            x_detached.requires_grad_(True)
            layer_output = layer(x_detached)
            jvp_final = torch.autograd.grad(
                layer_output, x_detached, 
                grad_outputs=v, 
                create_graph=False
            )[0]
            
            sigma_max_sq = torch.sum(jvp_final ** 2)
            return torch.sqrt(sigma_max_sq).item()
            
    except Exception as e:
        print(f"Warning: Could not compute singular value for layer: {e}")
        return 0.0

def analyze_spectral_properties(model, data_batch, device):
    """Analyze spectral properties of all layers in the model."""
    model.eval()
    
    if not hasattr(model, 'layers'):
        return {}
    
    data = data_batch[0].to(device)
    
    with torch.no_grad():
        x = data.view(data.size(0), -1)
        x = model.input_proj(x)
        activations = [x]
        
        for layer in model.layers:
            x = layer(x)
            activations.append(x)
    
    depth = len(model.layers)
    layers_to_analyze = [0, depth//4, depth//2, 3*depth//4, depth-1]
    layers_to_analyze = [i for i in layers_to_analyze if i < depth]
    
    spectral_data = {}
    
    for l_idx in layers_to_analyze:
        layer = model.layers[l_idx]
        layer_input = activations[l_idx]
        
        sigma_max = power_iteration_sigma_max(layer, layer_input)
        spectral_data[l_idx] = {
            'sigma_max': sigma_max,
            'layer_depth': l_idx
        }
        
        if hasattr(layer, 'log_lambda'):
            lambda_val = torch.exp(layer.log_lambda).item()
            spectral_data[l_idx]['lambda'] = lambda_val
    
    return spectral_data

def evaluate_model_depth(model_class, depth, train_loader, val_loader, device, 
                        epochs=10, lr=1e-3, weight_decay=0.01):
    """Evaluate if a model can be trained successfully at given depth."""
    from train import train_model
    
    try:
        model = model_class(depth=depth).to(device)
        
        print(f"Testing {model_class.__name__} at depth {depth}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        history, success = train_model(
            model, train_loader, val_loader, 
            epochs=epochs, lr=lr, weight_decay=weight_decay, 
            device=device, model_name=f"{model_class.__name__}_L{depth}"
        )
        
        if not success or history is None:
            return False, 0.0, None
        
        final_acc = history['val_acc'][-1]
        
        success = final_acc > 0.02
        
        return success, final_acc, history
        
    except Exception as e:
        print(f"Error training {model_class.__name__} at depth {depth}: {e}")
        return False, 0.0, None

def run_depth_sweep(model_classes, depths, train_loader, val_loader, device, 
                   epochs=10, num_seeds=3):
    """Run depth sweep experiment across multiple model types."""
    results = {}
    
    for model_name, model_class in model_classes.items():
        print(f"\n=== Depth Sweep for {model_name} ===")
        results[model_name] = {
            'max_depth': 0,
            'depths_tested': [],
            'success_rates': [],
            'final_accuracies': []
        }
        
        for depth in depths:
            print(f"\nTesting depth {depth}...")
            
            successes = 0
            accuracies = []
            
            for seed in range(num_seeds):
                torch.manual_seed(42 + seed)
                torch.cuda.manual_seed(42 + seed)
                
                success, acc, _ = evaluate_model_depth(
                    model_class, depth, train_loader, val_loader, device, 
                    epochs=epochs
                )
                
                if success:
                    successes += 1
                    accuracies.append(acc)
                
                print(f"  Seed {seed}: {'SUCCESS' if success else 'FAILED'} "
                      f"(Acc: {acc:.4f})")
            
            success_rate = successes / num_seeds
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            
            results[model_name]['depths_tested'].append(depth)
            results[model_name]['success_rates'].append(success_rate)
            results[model_name]['final_accuracies'].append(avg_accuracy)
            
            print(f"  Overall: {successes}/{num_seeds} successful "
                  f"(Rate: {success_rate:.2f}, Avg Acc: {avg_accuracy:.4f})")
            
            if success_rate >= 0.67:
                results[model_name]['max_depth'] = depth
            else:
                print(f"  Stopping depth sweep for {model_name} at depth {depth}")
                break
    
    return results

def plot_depth_results(results, save_path):
    """Plot maximum trainable depth comparison."""
    model_names = list(results.keys())
    max_depths = [results[name]['max_depth'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, max_depths, 
                   color=['#2E86AB', '#A23B72', '#F18F01'])
    
    plt.title('Maximum Trainable Depth Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model Type', fontsize=14)
    plt.ylabel('Maximum Stable Depth', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, depth in zip(bars, max_depths):
        if depth > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{depth}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved depth comparison plot to {save_path}")

def plot_spectral_analysis(spectral_history, model_name, save_path):
    """Plot spectral properties over training."""
    if not spectral_history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Spectral Analysis: {model_name}', fontsize=16, fontweight='bold')
    
    epochs = list(range(len(spectral_history)))
    layer_indices = list(spectral_history[0].keys()) if spectral_history else []
    
    ax1 = axes[0, 0]
    for l_idx in layer_indices:
        sigma_values = [spectral_history[epoch][l_idx]['sigma_max'] 
                       for epoch in epochs if l_idx in spectral_history[epoch]]
        if sigma_values:
            ax1.plot(epochs[:len(sigma_values)], sigma_values, 
                    label=f'Layer {l_idx}', marker='o', markersize=3)
    
    ax1.set_title('Maximum Singular Values')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('σ_max')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Isometry')
    
    ax2 = axes[0, 1]
    has_lambda = False
    for l_idx in layer_indices:
        if (spectral_history and 
            spectral_history[0].get(l_idx, {}).get('lambda') is not None):
            lambda_values = [spectral_history[epoch][l_idx]['lambda'] 
                           for epoch in epochs if l_idx in spectral_history[epoch]]
            if lambda_values:
                ax2.semilogy(epochs[:len(lambda_values)], lambda_values, 
                           label=f'Layer {l_idx}', marker='o', markersize=3)
                has_lambda = True
    
    if has_lambda:
        ax2.set_title('Adaptive Regularization Strength (λ)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('λ (log scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No λ values\n(Not AJC model)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Adaptive Regularization Strength (λ)')
    
    ax3 = axes[1, 0]
    for l_idx in layer_indices:
        sigma_values = [spectral_history[epoch][l_idx]['sigma_max'] 
                       for epoch in epochs if l_idx in spectral_history[epoch]]
        if sigma_values:
            deviations = [abs(s - 1.0) for s in sigma_values]
            ax3.semilogy(epochs[:len(deviations)], deviations, 
                        label=f'Layer {l_idx}', marker='o', markersize=3)
    
    ax3.set_title('Deviation from Isometry |σ_max - 1|')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('|σ_max - 1| (log scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    if spectral_history:
        final_epoch = spectral_history[-1]
        layers = list(final_epoch.keys())
        final_sigmas = [final_epoch[l]['sigma_max'] for l in layers]
        
        ax4.bar(range(len(layers)), final_sigmas, alpha=0.7)
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('Final Singular Values by Layer')
        ax4.set_xlabel('Layer Index')
        ax4.set_ylabel('σ_max')
        ax4.set_xticks(range(len(layers)))
        ax4.set_xticklabels([f'L{l}' for l in layers])
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved spectral analysis plot to {save_path}")

def comprehensive_evaluation(model_classes, train_loader, val_loader, device, 
                           save_dir='.research/iteration1/images'):
    """Run comprehensive evaluation of all models."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("=== Starting Comprehensive Evaluation ===")
    
    print("\n1. Running depth sweep experiment...")
    depths = [16, 32, 64, 128]  # Reduced for faster testing
    depth_results = run_depth_sweep(
        model_classes, depths, train_loader, val_loader, device, 
        epochs=5, num_seeds=2  # Reduced for faster testing
    )
    
    depth_plot_path = os.path.join(save_dir, 'depth_comparison.pdf')
    plot_depth_results(depth_results, depth_plot_path)
    
    print("\n2. Running spectral analysis...")
    fixed_depth = 32
    spectral_results = {}
    
    for model_name, model_class in model_classes.items():
        print(f"\nAnalyzing {model_name} at depth {fixed_depth}...")
        
        model = model_class(depth=fixed_depth).to(device)
        spectral_history = []
        
        data_batch = next(iter(val_loader))
        
        for epoch in range(5):  # Reduced for testing
            spectral_data = analyze_spectral_properties(model, data_batch, device)
            spectral_history.append(spectral_data)
            
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            for i, (data, target) in enumerate(train_loader):
                if i >= 10:  # Just a few batches
                    break
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target) + model.get_regularization_loss()
                loss.backward()
                optimizer.step()
        
        spectral_results[model_name] = spectral_history
        
        spectral_plot_path = os.path.join(save_dir, f'spectral_analysis_{model_name.replace("-", "_")}.pdf')
        plot_spectral_analysis(spectral_history, model_name, spectral_plot_path)
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to: {save_dir}")
    
    return depth_results, spectral_results

if __name__ == "__main__":
    from train import AJC_LGN, LGN_Residual, LGN_Vanilla
    from preprocess import get_cifar100_loaders, set_seed
    
    print("Testing evaluation functions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    train_loader, val_loader = get_cifar100_loaders(batch_size=32, test_run=True)
    
    model_classes = {
        'AJC-LGN': AJC_LGN,
        'LGN-Residual': LGN_Residual,
        'LGN-Vanilla': LGN_Vanilla
    }
    
    print("\nTesting spectral analysis...")
    model = AJC_LGN(depth=16).to(device)
    data_batch = next(iter(val_loader))
    spectral_data = analyze_spectral_properties(model, data_batch, device)
    print(f"Spectral data: {spectral_data}")
    
    print("\nEvaluation testing completed!")
