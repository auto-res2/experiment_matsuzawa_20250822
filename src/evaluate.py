"""
BEMeGA Evaluation Module
Implements evaluation logic and comparison with baseline methods
"""

import math
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

from train import BEMeGAAdapter, ProtoNetBaseline, build_random_dictionary, compute_support_stats
from train import mahalanobis_mixture_logits, spherical_kmeans_torch
from preprocess import SyntheticEpisodeGenerator, create_episode_batch


def simple_label_propagation(Zs: torch.Tensor, ys: torch.Tensor, Zq: torch.Tensor, 
                            logits_init: torch.Tensor, alpha: float = 0.5, 
                            k: int = 10, iters: int = 3) -> torch.Tensor:
    """Graph label propagation on query points with support centroids as anchors"""
    with torch.no_grad():
        C = logits_init.size(1)
        classes = torch.unique(ys)
        anchors = []
        for c in classes:
            anchors.append(Zs[ys == c].mean(0))
        A = torch.stack(anchors, dim=0)
        
        Z_all = torch.cat([A, Zq], dim=0)
        ZN = F.normalize(Z_all, dim=1)
        sim = ZN @ ZN.t()
        
        topk = min(k, sim.size(0) - 1)
        mask = torch.zeros_like(sim)
        idx = torch.topk(sim, k=topk + 1, dim=1).indices
        rows = torch.arange(sim.size(0), device=sim.device).unsqueeze(1).repeat(1, topk + 1)
        mask[rows, idx] = 1.0
        mask = mask * (1 - torch.eye(sim.size(0), device=sim.device))
        W = sim * mask
        W = F.relu(W)
        D = W.sum(dim=1, keepdim=True) + 1e-6
        S = W / D
        
        Y = torch.zeros(Z_all.size(0), C, device=Zq.device, dtype=Zq.dtype)
        Y[:C, :] = torch.eye(C, device=Zq.device, dtype=Zq.dtype)
        Y0q = F.softmax(logits_init, dim=1)
        Y[C:, :] = Y0q
        
        for _ in range(iters):
            Y = alpha * (S @ Y) + (1 - alpha) * Y
            Y[:C, :] = (1 - alpha) * torch.eye(C, device=Zq.device, dtype=Zq.dtype) + alpha * Y[:C, :]
        
        return Y[C:, :]


def evaluate_bemega(adapter: BEMeGAAdapter, episodes: List, device: str = "cpu", 
                   use_transduction: bool = False, risk_threshold: float = 0.5) -> Dict:
    """Evaluate BEMeGA adapter on episodes"""
    adapter.eval()
    accuracies = []
    transductive_calls = 0
    adapter_diagnostics = []
    
    with torch.no_grad():
        for support_data, support_labels, query_data, query_labels in episodes:
            support_data = support_data.to(device)
            support_labels = support_labels.to(device)
            query_data = query_data.to(device)
            query_labels = query_labels.to(device)
            
            stats = compute_support_stats(support_data, support_labels)
            adapter_out = adapter(stats)
            
            Zs_proj = support_data @ adapter_out.P * adapter_out.tau_P
            Zq_proj = query_data @ adapter_out.P * adapter_out.tau_P
            
            classes = torch.unique(support_labels).tolist()
            protos = {}
            
            for c in classes:
                class_data = Zs_proj[support_labels == c]
                if adapter_out.m_vec[c] == 1:
                    protos[c] = [class_data.mean(0)]
                else:
                    centers = spherical_kmeans_torch(class_data, adapter_out.m_vec[c])
                    protos[c] = [centers[i] for i in range(centers.size(0))]
            
            logits = mahalanobis_mixture_logits(Zq_proj, protos, adapter_out.cov_params, adapter_out.priors)
            
            if use_transduction and adapter_out.risk_hat > risk_threshold:
                transductive_calls += 1
                refined_probs = simple_label_propagation(Zs_proj, support_labels, Zq_proj, 
                                                       logits, alpha=adapter_out.lam)
                logits = torch.log(refined_probs + 1e-8)
            
            pred = logits.argmax(dim=1)
            acc = (pred == query_labels).float().mean().item()
            accuracies.append(acc)
            
            adapter_diagnostics.append({
                'risk_hat': adapter_out.risk_hat,
                'd': adapter_out.d,
                'r': adapter_out.r,
                'tau_P': adapter_out.tau_P,
                'm_total': sum(adapter_out.m_vec.values()),
                'silhouette': stats['silhouette'],
                'tw_tb_ratio': float(stats['tw_tb_ratio']),
                'trace_dispersion': float(stats['trace_dispersion'])
            })
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'accuracies': accuracies,
        'transductive_rate': transductive_calls / len(episodes),
        'diagnostics': adapter_diagnostics
    }


def evaluate_protonet(baseline: ProtoNetBaseline, episodes: List, device: str = "cpu") -> Dict:
    """Evaluate ProtoNet baseline on episodes"""
    baseline.eval()
    accuracies = []
    
    with torch.no_grad():
        for support_data, support_labels, query_data, query_labels in episodes:
            support_data = support_data.to(device)
            support_labels = support_labels.to(device)
            query_data = query_data.to(device)
            query_labels = query_labels.to(device)
            
            logits = baseline(support_data, support_labels, query_data)
            pred = logits.argmax(dim=1)
            acc = (pred == query_labels).float().mean().item()
            accuracies.append(acc)
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'accuracies': accuracies
    }


def run_k_mismatch_experiment(D: int = 128, device: str = "cpu") -> Dict:
    """Run k-mismatch robustness experiment"""
    print("Running k-mismatch robustness experiment...")
    
    dict_bank = build_random_dictionary(D, device=device)
    adapter = BEMeGAAdapter(D, dict_bank, device=device)
    baseline = ProtoNetBaseline(D, device=device)
    generator = SyntheticEpisodeGenerator(D=D, device=device)
    
    train_k = 5
    train_episodes = create_episode_batch(generator, "standard", N=5, k=train_k, q=15, batch_size=100)
    
    from train import train_bemega_adapter
    print("Training BEMeGA adapter...")
    train_bemega_adapter(adapter, train_episodes, num_epochs=50, device=device)
    
    results = {}
    test_k_values = [1, 2, 3, 5, 7, 10]
    
    for test_k in test_k_values:
        print(f"Testing with k={test_k}...")
        test_episodes = create_episode_batch(generator, "standard", N=5, k=test_k, q=15, batch_size=50)
        
        bemega_results = evaluate_bemega(adapter, test_episodes, device=device)
        protonet_results = evaluate_protonet(baseline, test_episodes, device=device)
        
        results[test_k] = {
            'bemega': bemega_results,
            'protonet': protonet_results
        }
    
    return results


def run_anisotropy_experiment(D: int = 128, device: str = "cpu") -> Dict:
    """Run anisotropy robustness experiment"""
    print("Running anisotropy robustness experiment...")
    
    dict_bank = build_random_dictionary(D, device=device)
    adapter = BEMeGAAdapter(D, dict_bank, device=device)
    baseline = ProtoNetBaseline(D, device=device)
    generator = SyntheticEpisodeGenerator(D=D, device=device)
    
    train_episodes = create_episode_batch(generator, "anisotropic", N=5, k=5, q=15, 
                                        batch_size=100, anisotropy_factor=3.0)
    
    from train import train_bemega_adapter
    print("Training BEMeGA adapter...")
    train_bemega_adapter(adapter, train_episodes, num_epochs=50, device=device)
    
    results = {}
    anisotropy_factors = [1.0, 2.0, 3.0, 5.0, 8.0]
    
    for factor in anisotropy_factors:
        print(f"Testing with anisotropy factor={factor}...")
        test_episodes = create_episode_batch(generator, "anisotropic", N=5, k=5, q=15, 
                                           batch_size=50, anisotropy_factor=factor)
        
        bemega_results = evaluate_bemega(adapter, test_episodes, device=device)
        protonet_results = evaluate_protonet(baseline, test_episodes, device=device)
        
        results[factor] = {
            'bemega': bemega_results,
            'protonet': protonet_results
        }
    
    return results


def run_domain_shift_experiment(D: int = 128, device: str = "cpu") -> Dict:
    """Run domain shift robustness experiment"""
    print("Running domain shift robustness experiment...")
    
    dict_bank = build_random_dictionary(D, device=device)
    adapter = BEMeGAAdapter(D, dict_bank, device=device)
    baseline = ProtoNetBaseline(D, device=device)
    generator = SyntheticEpisodeGenerator(D=D, device=device)
    
    train_episodes = create_episode_batch(generator, "standard", N=5, k=5, q=15, batch_size=100)
    
    from train import train_bemega_adapter
    print("Training BEMeGA adapter...")
    train_bemega_adapter(adapter, train_episodes, num_epochs=50, device=device)
    
    results = {}
    shift_factors = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    for factor in shift_factors:
        print(f"Testing with domain shift factor={factor}...")
        if factor == 0.0:
            test_episodes = create_episode_batch(generator, "standard", N=5, k=5, q=15, batch_size=50)
        else:
            test_episodes = create_episode_batch(generator, "domain_shift", N=5, k=5, q=15, 
                                               batch_size=50, shift_factor=factor)
        
        bemega_results = evaluate_bemega(adapter, test_episodes, device=device, 
                                       use_transduction=True, risk_threshold=0.5)
        protonet_results = evaluate_protonet(baseline, test_episodes, device=device)
        
        results[factor] = {
            'bemega': bemega_results,
            'protonet': protonet_results
        }
    
    return results


def plot_results(results: Dict, experiment_name: str, x_label: str, save_dir: str):
    """Plot experimental results and save as PDF"""
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x_values = sorted(results.keys())
    bemega_means = [results[x]['bemega']['mean_accuracy'] for x in x_values]
    bemega_stds = [results[x]['bemega']['std_accuracy'] for x in x_values]
    protonet_means = [results[x]['protonet']['mean_accuracy'] for x in x_values]
    protonet_stds = [results[x]['protonet']['std_accuracy'] for x in x_values]
    
    ax1.errorbar(x_values, bemega_means, yerr=bemega_stds, marker='o', label='BEMeGA', linewidth=2)
    ax1.errorbar(x_values, protonet_means, yerr=protonet_stds, marker='s', label='ProtoNet', linewidth=2)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{experiment_name} - Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    improvements = [(bemega_means[i] - protonet_means[i]) * 100 for i in range(len(x_values))]
    ax2.bar(range(len(x_values)), improvements, alpha=0.7)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title(f'{experiment_name} - BEMeGA Improvement')
    ax2.set_xticks(range(len(x_values)))
    ax2.set_xticklabels([str(x) for x in x_values])
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{experiment_name.lower().replace(" ", "_")}_results.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_diagnostics(results: Dict, experiment_name: str, save_dir: str):
    """Plot adapter diagnostics"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x_values = sorted(results.keys())
    
    risk_hats = []
    dimensions = []
    tau_values = []
    m_totals = []
    
    for x in x_values:
        diagnostics = results[x]['bemega']['diagnostics']
        risk_hats.append([d['risk_hat'] for d in diagnostics])
        dimensions.append([d['d'] for d in diagnostics])
        tau_values.append([d['tau_P'] for d in diagnostics])
        m_totals.append([d['m_total'] for d in diagnostics])
    
    axes[0, 0].boxplot(risk_hats, labels=[str(x) for x in x_values])
    axes[0, 0].set_title('Risk Prediction')
    axes[0, 0].set_ylabel('Risk Hat')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].boxplot(dimensions, labels=[str(x) for x in x_values])
    axes[0, 1].set_title('Intrinsic Dimension')
    axes[0, 1].set_ylabel('Dimension d')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].boxplot(tau_values, labels=[str(x) for x in x_values])
    axes[1, 0].set_title('Temperature Scaling')
    axes[1, 0].set_ylabel('tau_P')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].boxplot(m_totals, labels=[str(x) for x in x_values])
    axes[1, 1].set_title('Total Prototypes')
    axes[1, 1].set_ylabel('Total m')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{experiment_name.lower().replace(" ", "_")}_diagnostics.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing evaluation module on {device}")
    
    save_dir = "../.research/iteration1/images"
    os.makedirs(save_dir, exist_ok=True)
    
    k_results = run_k_mismatch_experiment(device=device)
    plot_results(k_results, "K-Mismatch Robustness", "Test k", save_dir)
    plot_diagnostics(k_results, "K-Mismatch Robustness", save_dir)
    
    print("Evaluation module test completed successfully!")
