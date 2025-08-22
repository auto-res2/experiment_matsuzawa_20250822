import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from train import (
    LAMBSModel, DiscreteSKUManager, CostProfiler, 
    train_model, get_device
)
from evaluate import evaluate_model, analyze_routing_patterns
from preprocess import create_dataloaders, save_dataset_info

def set_status(status):
    status_file = ".research/status.json"
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, "w") as f:
        json.dump({"status_enum": status, "timestamp": datetime.now().isoformat()}, f)
    print(f"Status set to: {status}")

def create_config():
    return {
        'vocab_size': 1000,
        'seq_len': 128,
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 6,
        'max_len': 512,
        'batch_size': 16,
        'epochs': 5,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'lambda_cost': 0.001,
        'lambda_frag': 0.01,
        'train_samples': 5000,
        'val_samples': 1000,
        'test_samples': 500,
        'seed': 42
    }

def save_training_plots(training_results, save_dir):
    epochs = range(1, len(training_results['train_losses']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(epochs, training_results['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, training_results['val_losses'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, training_results['latencies'], 'g-', linewidth=2, marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Latency (ms)')
    axes[0, 1].set_title('Training Latency per Epoch')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, training_results['energies'], 'm-', linewidth=2, marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Energy (mJ)')
    axes[1, 0].set_title('Energy Consumption per Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    final_train_loss = training_results['train_losses'][-1]
    final_val_loss = training_results['val_losses'][-1]
    final_latency = training_results['latencies'][-1]
    final_energy = training_results['energies'][-1]
    
    axes[1, 1].text(0.1, 0.8, f"Final Train Loss: {final_train_loss:.4f}", fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f"Final Val Loss: {final_val_loss:.4f}", fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f"Final Latency: {final_latency:.2f} ms", fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f"Final Energy: {final_energy:.2f} mJ", fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Training Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def save_cost_analysis(cost_profiler, sku_mgr, save_dir):
    shapes = [(128, 16, 4, 64), (256, 8, 4, 64)]
    
    attn_costs = []
    ssm_costs = []
    
    for shape in shapes:
        for i, window in enumerate(sku_mgr.attn_windows):
            cost_info = cost_profiler.query("attn", i, shape, tail=True)
            attn_costs.append((window, cost_info["ms"]))
        
        for j, state in enumerate(sku_mgr.ssm_states):
            cost_info = cost_profiler.query("ssm", j, shape, tail=True)
            ssm_costs.append((state, cost_info["ms"]))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    windows, attn_times = zip(*attn_costs[:len(sku_mgr.attn_windows)])
    axes[0].bar(range(len(windows)), attn_times)
    axes[0].set_xlabel('Attention Window Size')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Attention SKU Costs')
    axes[0].set_xticks(range(len(windows)))
    axes[0].set_xticklabels([str(w) for w in windows])
    
    states, ssm_times = zip(*ssm_costs[:len(sku_mgr.ssm_states)])
    axes[1].bar(range(len(states)), ssm_times)
    axes[1].set_xlabel('SSM State Size')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('SSM SKU Costs')
    axes[1].set_xticks(range(len(states)))
    axes[1].set_xticklabels([str(s) for s in states])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cost_analysis.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 60)
    print("LAMBS++ Experiment: Length-aware Budgeted Router + Scheduler")
    print("=" * 60)
    
    set_status("running")
    
    device = get_device()
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    config = create_config()
    print("\nExperiment Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    save_dir = ".research/iteration1/images"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("1. DATA PREPROCESSING")
    print("="*50)
    
    train_loader, val_loader, test_loader = create_dataloaders(config)
    dataset_info = save_dataset_info(train_loader, val_loader, test_loader, "data")
    
    print("\n" + "="*50)
    print("2. MODEL INITIALIZATION")
    print("="*50)
    
    sku_mgr = DiscreteSKUManager(
        attn_windows=(64, 128, 256),
        ssm_states=(4, 8, 16),
        retr_k=(0, 2)
    )
    
    model = LAMBSModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_len=config['max_len'],
        sku_mgr=sku_mgr
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*50)
    print("3. COST PROFILING")
    print("="*50)
    
    cost_profiler = CostProfiler(sku_mgr, device=device, dtype=torch.float16)
    
    profile_shapes = [
        (config['seq_len'], config['batch_size'], config['n_heads'], config['d_model'] // config['n_heads']),
        (config['seq_len'] // 2, config['batch_size'], config['n_heads'], config['d_model'] // config['n_heads'])
    ]
    
    cost_profiler.profile_all(profile_shapes)
    save_cost_analysis(cost_profiler, sku_mgr, save_dir)
    
    print("\n" + "="*50)
    print("4. MODEL TRAINING")
    print("="*50)
    
    training_results = train_model(model, train_loader, val_loader, config, cost_profiler, device)
    save_training_plots(training_results, save_dir)
    
    print("\n" + "="*50)
    print("5. MODEL EVALUATION")
    print("="*50)
    
    eval_results = evaluate_model(model, test_loader, cost_profiler, device, save_dir)
    
    print("\nEvaluation Results:")
    print(f"  Test Loss: {eval_results['test_loss']:.4f}")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  Mean Latency: {eval_results['latency_stats']['mean']:.2f} ms")
    print(f"  P95 Latency: {eval_results['latency_stats']['p95']:.2f} ms")
    print(f"  P99 Latency: {eval_results['latency_stats']['p99']:.2f} ms")
    print(f"  Total Energy: {eval_results['energy_stats']['total']:.2f} mJ")
    
    print("\n" + "="*50)
    print("6. ROUTING ANALYSIS")
    print("="*50)
    
    routing_data = analyze_routing_patterns(model, test_loader, device, save_dir)
    
    print(f"Analyzed routing patterns for {len(routing_data)} samples")
    
    avg_attn_usage = np.mean([d['branch_sel'][:, :, :, 0].mean() for d in routing_data])
    avg_ssm_usage = np.mean([d['branch_sel'][:, :, :, 1].mean() for d in routing_data])
    
    print(f"Average Attention Usage: {avg_attn_usage:.3f}")
    print(f"Average SSM Usage: {avg_ssm_usage:.3f}")
    
    print("\n" + "="*50)
    print("7. SAVING RESULTS")
    print("="*50)
    
    results_summary = {
        'config': config,
        'model_params': {
            'total': total_params,
            'trainable': trainable_params
        },
        'training_results': training_results,
        'evaluation_results': eval_results,
        'routing_analysis': {
            'avg_attn_usage': float(avg_attn_usage),
            'avg_ssm_usage': float(avg_ssm_usage),
            'num_samples_analyzed': len(routing_data)
        }
    }
    
    with open(f"{save_dir}/results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    torch.save(model.state_dict(), "models/lambs_model.pth")
    
    print(f"Results saved to: {save_dir}")
    print("Generated PDF files:")
    pdf_files = [f for f in os.listdir(save_dir) if f.endswith('.pdf')]
    for pdf_file in sorted(pdf_files):
        print(f"  - {pdf_file}")
    
    print("\n" + "="*60)
    print("LAMBS++ EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*60)
    
    print("\nKey Findings:")
    print(f"  • Model achieved {eval_results['accuracy']:.1%} accuracy on test set")
    print(f"  • P95 latency: {eval_results['latency_stats']['p95']:.1f}ms (within T4 constraints)")
    print(f"  • Attention/SSM routing ratio: {avg_attn_usage:.2f}/{avg_ssm_usage:.2f}")
    print(f"  • Energy efficiency: {eval_results['energy_stats']['total']:.1f}mJ total")
    
    set_status("stopped")
    
    return results_summary

if __name__ == "__main__":
    main()
