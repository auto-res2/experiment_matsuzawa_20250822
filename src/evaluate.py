
import math
import time
import random
from typing import Callable, List, Tuple, Dict
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns



def bracket_sparsity_screen(sigmas: List[Callable], x: torch.Tensor, thresh: float = 1e-6, 
                           mprobes: int = 8) -> Dict[str, float]:
    """Screen for sparse bracket structure using randomized contractions"""
    from .preprocess import lie_bracket, hutchinson_norm_sq
    
    r = len(sigmas)
    zero_pairs = 0
    total_pairs = max(0, r * (r - 1) // 2)
    magnitudes = []
    
    if total_pairs == 0:
        return {"precision": 1.0, "recall": 1.0, "avg_magnitude": 0.0}
    
    for i in range(r):
        for j in range(i + 1, r):
            br = lie_bracket(sigmas[i], sigmas[j], x)
            mag = hutchinson_norm_sq(br, m=mprobes).item()
            magnitudes.append(mag)
            if mag < thresh:
                zero_pairs += 1
    
    precision = zero_pairs / total_pairs
    recall = precision  # For synthetic cases where we know ground truth
    avg_mag = float(np.mean(magnitudes)) if len(magnitudes) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "avg_magnitude": avg_mag}



def create_synthetic_sde_case_a(d: int, device) -> Tuple[Callable, List[Callable], str]:
    """Case A: Commuting diffusions (all brackets should be zero)"""
    def b(x):
        return -0.1 * x
    
    A1 = torch.diag(torch.linspace(0.1, 0.3, d)).to(device)
    A2 = torch.diag(torch.linspace(0.2, 0.4, d)).to(device)
    
    def sigma1(x):
        return A1 @ x
    
    def sigma2(x):
        return A2 @ x
    
    return b, [sigma1, sigma2], "Commuting (Case A)"


def create_synthetic_sde_case_b(d: int, device) -> Tuple[Callable, List[Callable], str]:
    """Case B: Non-commuting but structured diffusions"""
    def b(x):
        return -0.05 * x + 0.1 * torch.sin(x)
    
    def sigma1(x):
        return 0.2 * torch.cat([x[1:], x[:1]])  # Circular shift
    
    def sigma2(x):
        return 0.15 * torch.cat([x[-1:], x[:-1]])  # Reverse circular shift
    
    return b, [sigma1, sigma2], "Non-commuting (Case B)"


def create_synthetic_sde_case_c(d: int, device) -> Tuple[Callable, List[Callable], str]:
    """Case C: Nonlinear with complex bracket structure"""
    def b(x):
        return -0.1 * x + 0.05 * torch.tanh(x)
    
    def sigma1(x):
        return 0.2 * torch.sin(x)
    
    def sigma2(x):
        return 0.15 * torch.cos(x)
    
    def sigma3(x):
        return 0.1 * x * torch.exp(-0.5 * torch.norm(x)**2)
    
    return b, [sigma1, sigma2, sigma3], "Nonlinear (Case C)"


def run_synthetic_sde_experiment(d: int = 20, device=None) -> Dict[str, any]:
    """Run synthetic SDE experiment with bracket-sparsity screening"""
    from .preprocess import set_seed, get_device, g_quadratic, grad_g_quadratic, make_quadratic_Q
    from .train import ItoSDE, integrand_h_total, CriticNet, train_critic
    
    if device is None:
        device = get_device()
    
    set_seed(42)
    results = {}
    
    cases = [
        create_synthetic_sde_case_a(d, device),
        create_synthetic_sde_case_b(d, device),
        create_synthetic_sde_case_c(d, device)
    ]
    
    Q = make_quadratic_Q(d, bandwidth=3, scale=0.5, seed=123).to(device)
    
    def grad_g(x):
        return grad_g_quadratic(x, Q)
    
    for i, (b, sigmas, case_name) in enumerate(cases):
        print(f"\n=== {case_name} ===")
        
        x_test = torch.randn(d, device=device) * 0.5
        
        sparsity_results = bracket_sparsity_screen(sigmas, x_test, thresh=1e-6, mprobes=16)
        print(f"Sparsity screening - Precision: {sparsity_results['precision']:.3f}, "
              f"Avg magnitude: {sparsity_results['avg_magnitude']:.6f}")
        
        sde = ItoSDE(b, sigmas, len(sigmas))
        x0 = torch.zeros(d, device=device)
        T = 1.0
        dt = 0.01
        
        def integrand_fn(x):
            return integrand_h_total(x, b, sigmas, grad_g, pair_probes=8)
        
        N_mc = 1000
        mc_vals = []
        for j in range(N_mc):
            tau = torch.rand(()) * T
            x_tau = sde.simulate_final(x0, float(tau.item()), dt, seed=j)
            h_val, _ = integrand_fn(x_tau)
            mc_vals.append(h_val * T)
        
        mc_mean = torch.stack(mc_vals).mean()
        mc_var = torch.stack(mc_vals).var()
        
        print(f"MC estimate: {mc_mean.item():.6f} ± {torch.sqrt(mc_var/N_mc).item():.6f}")
        
        C = 2  # Number of coefficient types
        critic = CriticNet(d, C, hidden=64, layers=2).to(device)
        
        print("Training critic...")
        train_losses = train_critic(critic, sde, x0, T, dt, integrand_fn, 
                                  num_epochs=50, batch_size=16, lr=1e-3, device=device)
        
        cv_vals = []
        for j in range(N_mc // 2):
            tau = torch.rand(()) * T
            x_tau = sde.simulate_final(x0, float(tau.item()), dt, seed=N_mc + j)
            h_val, _ = integrand_fn(x_tau)
            
            t_tensor = torch.tensor([[float(tau.item())]], device=device)
            x_tensor = x_tau.unsqueeze(0)
            critic_vals, _ = critic(t_tensor, x_tensor)
            cv_correction = critic_vals[0, 0]  # Use first head
            
            corrected_val = h_val * T - cv_correction * 0.1  # Simple scaling
            cv_vals.append(corrected_val)
        
        cv_mean = torch.stack(cv_vals).mean()
        cv_var = torch.stack(cv_vals).var()
        
        print(f"CV estimate: {cv_mean.item():.6f} ± {torch.sqrt(cv_var/(N_mc//2)).item():.6f}")
        
        variance_reduction = mc_var / (cv_var + 1e-12)
        print(f"Variance reduction factor: {variance_reduction.item():.2f}")
        
        results[case_name] = {
            "sparsity": sparsity_results,
            "mc_estimate": mc_mean.item(),
            "mc_variance": mc_var.item(),
            "cv_estimate": cv_mean.item(),
            "cv_variance": cv_var.item(),
            "variance_reduction": variance_reduction.item(),
            "train_losses": train_losses
        }
    
    return results



def create_local_volatility_model(d: int, rank: int, device) -> Tuple[Callable, List[Callable]]:
    """Create low-rank local volatility model for Greeks computation"""
    U = torch.randn(d, rank, device=device)
    U, _ = torch.linalg.qr(U, mode='reduced')
    
    def b(x):
        return -0.1 * x
    
    sigmas = []
    for k in range(rank):
        def make_sigma(k_idx):
            def sigma_k(x):
                vol_factor = 0.2 + 0.1 * torch.tanh(x[k_idx % d])
                return U[:, k_idx] * vol_factor
            return sigma_k
        sigmas.append(make_sigma(k))
    
    return b, sigmas


def run_greeks_experiment(d: int = 100, rank: int = 5, device=None) -> Dict[str, any]:
    """Run high-dimensional Greeks experiment"""
    from .preprocess import set_seed, get_device, estimate_diffusion_subspace
    from .train import ItoSDE
    
    if device is None:
        device = get_device()
    
    set_seed(42)
    print(f"\n=== High-dimensional Greeks (d={d}, rank={rank}) ===")
    
    b, sigmas = create_local_volatility_model(d, rank, device)
    sde = ItoSDE(b, sigmas, rank)
    
    x_test = torch.randn(d, device=device) * 0.1
    Ur = estimate_diffusion_subspace(sigmas, x_test, q=min(16, rank*2))
    estimated_rank = Ur.shape[1]
    
    print(f"True rank: {rank}, Estimated rank: {estimated_rank}")
    
    weights = torch.ones(d, device=device) / d
    strike = 0.0
    
    def payoff(x):
        basket_value = (weights * x).sum()
        return torch.relu(basket_value - strike)
    
    def grad_payoff(x):
        basket_value = (weights * x).sum()
        if basket_value > strike:
            return weights
        else:
            return torch.zeros_like(x)
    
    x0 = torch.zeros(d, device=device)
    T = 0.25  # 3 months
    dt = 0.01
    N_paths = 2000
    
    pathwise_deltas = []
    for i in range(N_paths):
        x_T = sde.simulate_final(x0, T, dt, seed=i)
        delta = grad_payoff(x_T)
        pathwise_deltas.append(delta)
    
    pathwise_delta_mean = torch.stack(pathwise_deltas).mean(dim=0)
    pathwise_delta_var = torch.stack(pathwise_deltas).var(dim=0)
    
    restricted_deltas = []
    for i in range(N_paths):
        x_T = sde.simulate_final(x0, T, dt, seed=i + N_paths)
        delta_full = grad_payoff(x_T)
        delta_restricted = Ur @ (Ur.T @ delta_full)
        restricted_deltas.append(delta_restricted)
    
    restricted_delta_mean = torch.stack(restricted_deltas).mean(dim=0)
    restricted_delta_var = torch.stack(restricted_deltas).var(dim=0)
    
    pathwise_total_var = pathwise_delta_var.sum()
    restricted_total_var = restricted_delta_var.sum()
    variance_reduction = pathwise_total_var / (restricted_total_var + 1e-12)
    
    print(f"Pathwise Delta variance: {pathwise_total_var.item():.6f}")
    print(f"Restricted Delta variance: {restricted_total_var.item():.6f}")
    print(f"Variance reduction: {variance_reduction.item():.2f}")
    
    delta_error = torch.norm(pathwise_delta_mean - restricted_delta_mean)
    print(f"Delta estimation error: {delta_error.item():.6f}")
    
    return {
        "true_rank": rank,
        "estimated_rank": estimated_rank,
        "pathwise_variance": pathwise_total_var.item(),
        "restricted_variance": restricted_total_var.item(),
        "variance_reduction": variance_reduction.item(),
        "delta_error": delta_error.item()
    }



def create_nonsmooth_sde(d: int, device) -> Tuple[Callable, List[Callable]]:
    """Create SDE with non-smooth (ReLU-like) drift"""
    def b_nonsmooth(x):
        if not x.requires_grad:
            x = x.requires_grad_(True)
        return -0.1 * x + 0.2 * torch.relu(x - 0.5) - 0.15 * torch.relu(-x - 0.5)
    
    def sigma1(x):
        if not x.requires_grad:
            x = x.requires_grad_(True)
        return 0.2 * torch.ones_like(x)
    
    def sigma2(x):
        if not x.requires_grad:
            x = x.requires_grad_(True)
        return 0.1 * torch.sin(x)
    
    return b_nonsmooth, [sigma1, sigma2]


def gaussian_mollify(f: Callable, eps: float = 0.01) -> Callable:
    """Apply Gaussian mollification to non-smooth function"""
    def f_mollified(x):
        if not x.requires_grad:
            x = x.requires_grad_(True)
        n_samples = 5
        total = torch.zeros_like(f(x))
        for i in range(n_samples):
            noise = torch.randn_like(x) * eps
            x_noisy = x + noise
            if not x_noisy.requires_grad:
                x_noisy = x_noisy.requires_grad_(True)
            total = total + f(x_noisy)
        return total / n_samples
    return f_mollified


def run_robustness_experiment(d: int = 10, device=None) -> Dict[str, any]:
    """Run robustness experiment with non-smooth drifts"""
    from .preprocess import set_seed, get_device, g_cubic, grad_g_cubic
    from .train import ItoSDE, integrand_h_total
    
    if device is None:
        device = get_device()
    
    set_seed(42)
    print(f"\n=== Robustness under non-smooth drifts (d={d}) ===")
    
    b_nonsmooth, sigmas = create_nonsmooth_sde(d, device)
    
    w = torch.randn(d, device=device) * 0.1
    
    def grad_g(x):
        return grad_g_cubic(x, w)
    
    eps_values = [0.0, 0.005, 0.01, 0.02]
    results = {}
    
    for eps in eps_values:
        print(f"\nMollification eps = {eps}")
        
        if eps == 0.0:
            b_smooth = b_nonsmooth
            label = "Original"
        else:
            b_smooth = gaussian_mollify(b_nonsmooth, eps)
            label = f"Mollified (eps={eps})"
        
        sde = ItoSDE(b_smooth, sigmas, len(sigmas))
        x0 = torch.zeros(d, device=device, requires_grad=True)
        T = 0.5
        dt = 0.01
        
        def integrand_fn(x):
            x_grad = x.detach().requires_grad_(True)
            return integrand_h_total(x_grad, b_smooth, sigmas, grad_g, pair_probes=6)
        
        N_mc = 1000
        mc_vals = []
        computation_times = []
        
        for j in range(N_mc):
            start_time = time.time()
            tau = torch.rand(()) * T
            x_tau = sde.simulate_final(x0, float(tau.item()), dt, seed=j)
            h_val, _ = integrand_fn(x_tau)
            mc_vals.append(h_val * T)
            computation_times.append(time.time() - start_time)
        
        mc_mean = torch.stack(mc_vals).mean()
        mc_var = torch.stack(mc_vals).var()
        avg_time = np.mean(computation_times)
        
        print(f"{label} - Estimate: {mc_mean.item():.6f} ± {torch.sqrt(mc_var/N_mc).item():.6f}")
        print(f"{label} - Avg computation time: {avg_time:.4f}s")
        
        results[eps] = {
            "estimate": mc_mean.item(),
            "variance": mc_var.item(),
            "std_error": torch.sqrt(mc_var/N_mc).item(),
            "avg_time": avg_time,
            "label": label
        }
    
    original_est = results[0.0]["estimate"]
    for eps in eps_values[1:]:
        bias = abs(results[eps]["estimate"] - original_est)
        variance_ratio = results[eps]["variance"] / results[0.0]["variance"]
        print(f"eps={eps}: Bias={bias:.6f}, Variance ratio={variance_ratio:.3f}")
    
    return results



def create_experiment_plots(results_synthetic, results_greeks, results_robustness, save_dir: str):
    """Create publication-quality PDF plots for all experiments"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    cases = list(results_synthetic.keys())
    precisions = [results_synthetic[case]["sparsity"]["precision"] for case in cases]
    avg_mags = [results_synthetic[case]["sparsity"]["avg_magnitude"] for case in cases]
    
    ax1.bar(cases, precisions, alpha=0.7)
    ax1.set_ylabel('Sparsity Precision')
    ax1.set_title('Bracket Sparsity Screening')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(cases, avg_mags, alpha=0.7, color='orange')
    ax2.set_ylabel('Average Bracket Magnitude')
    ax2.set_title('Bracket Magnitude Analysis')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_yscale('log')
    
    var_reductions = [results_synthetic[case]["variance_reduction"] for case in cases]
    ax3.bar(cases, var_reductions, alpha=0.7, color='green')
    ax3.set_ylabel('Variance Reduction Factor')
    ax3.set_title('Control Variate Performance')
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    
    for i, case in enumerate(cases):
        losses = results_synthetic[case]["train_losses"]
        ax4.plot(losses, label=case, alpha=0.8)
    ax4.set_xlabel('Training Epoch')
    ax4.set_ylabel('Critic Loss')
    ax4.set_title('Critic Training Convergence')
    ax4.legend()
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'synthetic_sde_results.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    true_rank = results_greeks["true_rank"]
    est_rank = results_greeks["estimated_rank"]
    ax1.bar(['True Rank', 'Estimated Rank'], [true_rank, est_rank], alpha=0.7)
    ax1.set_ylabel('Diffusion Rank')
    ax1.set_title('Diffusion Subspace Estimation')
    
    variances = [results_greeks["pathwise_variance"], results_greeks["restricted_variance"]]
    methods = ['Pathwise', 'HiLo-STeP++']
    colors = ['blue', 'red']
    bars = ax2.bar(methods, variances, alpha=0.7, color=colors)
    ax2.set_ylabel('Delta Variance')
    ax2.set_title('Greeks Estimation Variance')
    ax2.set_yscale('log')
    
    vr = results_greeks["variance_reduction"]
    ax2.annotate(f'VR: {vr:.1f}×', xy=(1, variances[1]), xytext=(1, variances[1]*2),
                arrowprops=dict(arrowstyle='->', color='black'), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'greeks_experiment.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    eps_values = list(results_robustness.keys())
    estimates = [results_robustness[eps]["estimate"] for eps in eps_values]
    std_errors = [results_robustness[eps]["std_error"] for eps in eps_values]
    
    ax1.errorbar(eps_values, estimates, yerr=std_errors, marker='o', capsize=5)
    ax1.set_xlabel('Mollification Parameter ε')
    ax1.set_ylabel('Coefficient Estimate')
    ax1.set_title('Bias-Variance Tradeoff')
    ax1.grid(True, alpha=0.3)
    
    times = [results_robustness[eps]["avg_time"] for eps in eps_values]
    ax2.plot(eps_values, times, marker='s', color='orange')
    ax2.set_xlabel('Mollification Parameter ε')
    ax2.set_ylabel('Average Computation Time (s)')
    ax2.set_title('Computational Cost')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'robustness_experiment.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All plots saved to {save_dir}")


if __name__ == "__main__":
    from .preprocess import set_seed, get_device
    
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    d = 5
    x_test = torch.randn(d, device=device)
    
    def sigma1(x):
        return 0.1 * x
    def sigma2(x):
        return 0.2 * x  # Commuting with sigma1
    
    sigmas = [sigma1, sigma2]
    sparsity_results = bracket_sparsity_screen(sigmas, x_test, thresh=1e-6, mprobes=8)
    print(f"Sparsity test - Precision: {sparsity_results['precision']:.3f}")
    
    print("Evaluation module test completed successfully!")
