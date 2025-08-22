import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from train import NVMLPowerSampler, Timer

def evaluate_model(model, test_loader, cost_profiler, device, save_dir):
    model.eval()
    model = model.to(device)
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    latencies = []
    energies = []
    routing_stats = {
        'attn_usage': [],
        'ssm_usage': [],
        'gate_values': [],
        'fragmentation_scores': []
    }
    
    timer = Timer()
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            
            with NVMLPowerSampler() as ps:
                t0 = time.time()
                
                def forward_fn():
                    return model(x)
                
                batch_latency = timer.time_ms(forward_fn)
                logits, router_outputs = forward_fn()
                
                t1 = time.time()
                batch_energy = ps.energy_mJ(t0, t1) or 0.0
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            test_loss += loss.item()
            
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            
            latencies.append(batch_latency)
            energies.append(batch_energy)
            
            for r in router_outputs:
                branch_sel = r["branch_sel"]
                attn_sku_sel = r["attn_sku_sel"]
                ssm_sku_sel = r["ssm_sku_sel"]
                gates = r["gates"]
                
                attn_usage = branch_sel[:, :, :, 0].mean().item()
                ssm_usage = branch_sel[:, :, :, 1].mean().item()
                
                routing_stats['attn_usage'].append(attn_usage)
                routing_stats['ssm_usage'].append(ssm_usage)
                routing_stats['gate_values'].append(gates.mean().item())
                
                if branch_sel.size(1) > 1:
                    frag = (branch_sel[:, 1:] - branch_sel[:, :-1]).abs().sum(dim=-1).mean().item()
                    routing_stats['fragmentation_scores'].append(frag)
    
    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    
    latency_stats = {
        'mean': np.mean(latencies),
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99)
    }
    
    energy_stats = {
        'mean': np.mean(energies),
        'total': np.sum(energies)
    }
    
    results = {
        'test_loss': avg_test_loss,
        'accuracy': accuracy,
        'latency_stats': latency_stats,
        'energy_stats': energy_stats,
        'routing_stats': routing_stats
    }
    
    save_evaluation_plots(results, save_dir)
    
    return results

def save_evaluation_plots(results, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(results['routing_stats']['attn_usage'], bins=20, alpha=0.7, label='Attention')
    axes[0, 0].hist(results['routing_stats']['ssm_usage'], bins=20, alpha=0.7, label='SSM')
    axes[0, 0].set_xlabel('Usage Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Branch Usage Distribution')
    axes[0, 0].legend()
    
    axes[0, 1].plot(results['routing_stats']['gate_values'])
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Gate Value')
    axes[0, 1].set_title('Gate Values Over Time')
    
    if results['routing_stats']['fragmentation_scores']:
        axes[1, 0].plot(results['routing_stats']['fragmentation_scores'])
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Fragmentation Score')
        axes[1, 0].set_title('Routing Fragmentation')
    
    latency_data = [results['latency_stats']['p50'], results['latency_stats']['p95'], results['latency_stats']['p99']]
    axes[1, 1].bar(['P50', 'P95', 'P99'], latency_data)
    axes[1, 1].set_ylabel('Latency (ms)')
    axes[1, 1].set_title('Latency Percentiles')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/evaluation_results.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.text(0.1, 0.8, f"Test Loss: {results['test_loss']:.4f}", fontsize=14, transform=ax.transAxes)
    ax.text(0.1, 0.7, f"Accuracy: {results['accuracy']:.4f}", fontsize=14, transform=ax.transAxes)
    ax.text(0.1, 0.6, f"Mean Latency: {results['latency_stats']['mean']:.2f} ms", fontsize=14, transform=ax.transAxes)
    ax.text(0.1, 0.5, f"P95 Latency: {results['latency_stats']['p95']:.2f} ms", fontsize=14, transform=ax.transAxes)
    ax.text(0.1, 0.4, f"Total Energy: {results['energy_stats']['total']:.2f} mJ", fontsize=14, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('LAMBS++ Evaluation Summary', fontsize=16, fontweight='bold')
    
    plt.savefig(f"{save_dir}/evaluation_summary.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_routing_patterns(model, test_loader, device, save_dir):
    model.eval()
    model = model.to(device)
    
    all_routing_data = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if batch_idx >= 10:
                break
            
            x = x.to(device)
            _, router_outputs = model(x)
            
            for layer_idx, r in enumerate(router_outputs):
                branch_sel = r["branch_sel"].cpu().numpy()
                attn_sku_sel = r["attn_sku_sel"].cpu().numpy()
                ssm_sku_sel = r["ssm_sku_sel"].cpu().numpy()
                gates = r["gates"].cpu().numpy()
                
                all_routing_data.append({
                    'batch': batch_idx,
                    'layer': layer_idx,
                    'branch_sel': branch_sel,
                    'attn_sku_sel': attn_sku_sel,
                    'ssm_sku_sel': ssm_sku_sel,
                    'gates': gates
                })
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    layer_attn_usage = []
    layer_ssm_usage = []
    for layer in range(len(router_outputs)):
        layer_data = [d for d in all_routing_data if d['layer'] == layer]
        attn_usage = np.mean([d['branch_sel'][:, :, :, 0].mean() for d in layer_data])
        ssm_usage = np.mean([d['branch_sel'][:, :, :, 1].mean() for d in layer_data])
        layer_attn_usage.append(attn_usage)
        layer_ssm_usage.append(ssm_usage)
    
    axes[0, 0].plot(layer_attn_usage, 'o-', label='Attention', linewidth=2, markersize=6)
    axes[0, 0].plot(layer_ssm_usage, 's-', label='SSM', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Usage Probability')
    axes[0, 0].set_title('Branch Usage by Layer')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    sample_data = all_routing_data[0]
    B, L, H = sample_data['branch_sel'].shape[:3]
    attn_heatmap = sample_data['branch_sel'][0, :, :, 0]
    im1 = axes[0, 1].imshow(attn_heatmap.T, aspect='auto', cmap='viridis')
    axes[0, 1].set_xlabel('Sequence Position')
    axes[0, 1].set_ylabel('Head')
    axes[0, 1].set_title('Attention Usage Heatmap (Sample)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    sku_usage = np.zeros(len(model.sku.attn_windows))
    for d in all_routing_data:
        for i in range(len(model.sku.attn_windows)):
            sku_usage[i] += d['attn_sku_sel'][:, :, :, i].mean()
    sku_usage /= len(all_routing_data)
    
    axes[1, 0].bar(range(len(model.sku.attn_windows)), sku_usage)
    axes[1, 0].set_xlabel('Attention Window SKU')
    axes[1, 0].set_ylabel('Usage Probability')
    axes[1, 0].set_title('Attention SKU Usage')
    axes[1, 0].set_xticks(range(len(model.sku.attn_windows)))
    axes[1, 0].set_xticklabels([str(w) for w in model.sku.attn_windows])
    
    gate_evolution = []
    for batch_idx in range(min(10, len(all_routing_data) // len(router_outputs))):
        batch_gates = []
        for layer in range(len(router_outputs)):
            data_idx = batch_idx * len(router_outputs) + layer
            if data_idx < len(all_routing_data):
                batch_gates.append(all_routing_data[data_idx]['gates'].mean())
        if batch_gates:
            gate_evolution.append(np.mean(batch_gates))
    
    axes[1, 1].plot(gate_evolution, 'o-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Batch')
    axes[1, 1].set_ylabel('Average Gate Value')
    axes[1, 1].set_title('Gate Evolution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/routing_analysis.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    return all_routing_data
