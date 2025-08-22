
import math
import time
import random
from typing import Callable, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp



class ItoSDE:
    """Euler-Maruyama SDE simulator for Itô SDEs"""
    
    def __init__(self, b: Callable, sigmas: List[Callable], r: int):
        self.b = b
        self.sigmas = sigmas
        self.r = r

    def sigma_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Build diffusion matrix S(x) = [sigma_1(x) ... sigma_r(x)]"""
        if self.r == 0:
            return torch.zeros(x.numel(), 0, device=x.device, dtype=x.dtype)
        return torch.stack([sj(x) for sj in self.sigmas], dim=1)  # d×r

    def simulate_to_time(self, x0: torch.Tensor, tau: float, dt: float, 
                        seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate SDE from x0 to time tau
        Returns: x_tau, last_dW_sum (r,), total_steps
        """
        torch.manual_seed(seed)
        x = x0.clone()
        steps = max(1, int(math.ceil(tau / dt)))
        d = x0.numel()
        last_dW_sum = torch.zeros(self.r, device=x0.device, dtype=x0.dtype)
        
        for k in range(steps):
            t_cur = min((k + 1) * dt, tau)
            h = dt if k < steps - 1 else (tau - k * dt)
            drift = self.b(x)
            
            if self.r > 0:
                S = self.sigma_matrix(x)
                dW = torch.randn(self.r, device=x.device, dtype=x.dtype) * math.sqrt(h)
                x = x + drift * h + S @ dW
                last_dW_sum = last_dW_sum + dW
            else:
                x = x + drift * h
                
        return x, last_dW_sum, torch.tensor(steps)

    def simulate_final(self, x0: torch.Tensor, T: float, dt: float, seed: int = 0) -> torch.Tensor:
        """Simulate SDE to final time T"""
        xT, _, _ = self.simulate_to_time(x0, T, dt, seed)
        return xT



def integrand_h_total(x: torch.Tensor, b: Callable, sigmas: List[Callable], 
                     grad_g: Callable[[torch.Tensor], torch.Tensor],
                     pair_probes: int = 4) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute total integrand for weak Taylor coefficients
    h_total = sum_i grad_g(x)·[b, sigma_i](x) + randomized estimate of sum_{i<j} grad_g(x)·[sigma_i, sigma_j](x)
    """
    from .preprocess import lie_bracket
    
    ggrad = grad_g(x)
    r = len(sigmas)
    
    h1 = torch.zeros((), device=x.device, dtype=x.dtype)
    for i in range(r):
        br = lie_bracket(b, sigmas[i], x)
        h1 = h1 + (ggrad * br).sum()
    
    h2 = torch.zeros((), device=x.device, dtype=x.dtype)
    if r >= 2 and pair_probes > 0:
        for _ in range(pair_probes):
            alpha = torch.randn(r, device=x.device, dtype=x.dtype)
            beta = torch.randn(r, device=x.device, dtype=x.dtype)
            
            def V(z):
                return sum(alpha[j] * sigmas[j](z) for j in range(r))
            def W(z):
                return sum(beta[j] * sigmas[j](z) for j in range(r))
            
            br_vw = lie_bracket(V, W, x)
            h2 = h2 + (ggrad * br_vw).sum()
        h2 = h2 / pair_probes
    
    h_total = h1 + h2
    parts = {"h_bsigma": h1.detach(), "h_sigmasigma": h2.detach()}
    return h_total, parts



class CriticNet(nn.Module):
    """Multi-head critic network with shared trunk for coefficient-specific control variates"""
    
    def __init__(self, d: int, C: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        mods = [nn.Linear(d + 1, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.trunk = nn.Sequential(*mods)
        self.val_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(C)])
        self.grad_heads = nn.ModuleList([nn.Linear(hidden, d) for _ in range(C)])

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning values and gradients for all heads"""
        h = self.trunk(torch.cat([t, x], dim=-1))
        vals = torch.cat([hd(h) for hd in self.val_heads], dim=-1)  # N×C
        grads = [gh(h) for gh in self.grad_heads]                   # list of N×d
        return vals, grads


def poisson_residual_loss(critic: CriticNet, batch_t: torch.Tensor, batch_x: torch.Tensor,
                         h_targets: torch.Tensor, b: Callable, Ur: torch.Tensor = None,
                         head_indices: List[int] = None) -> torch.Tensor:
    """Poisson residual loss: L c ≈ b·∇c (diffusion term omitted) and match to h"""
    vals, grads_list = critic(batch_t, batch_x)
    C = h_targets.shape[1]
    loss = torch.zeros((), device=batch_x.device)
    
    for ci in range(C):
        if head_indices is not None:
            hi = head_indices[ci]
        else:
            hi = ci
        grads = grads_list[hi]
        
        if Ur is not None:
            grads = grads @ (Ur @ Ur.T)
        
        if batch_x.ndim == 2:
            drift_term = torch.stack([b(batch_x[i]) for i in range(batch_x.shape[0])], dim=0)
        else:
            drift_term = b(batch_x)
            
        residual = (drift_term * grads).sum(dim=-1, keepdim=True) - h_targets[:, ci:ci+1]
        loss = loss + (residual.pow(2).mean())
    
    return loss / C



def sample_prior_states(d: int, N: int, mean: torch.Tensor, cov_diag: torch.Tensor, device) -> torch.Tensor:
    """Sample states from Gaussian prior for node selection"""
    z = torch.randn(N, d, device=device) * torch.sqrt(cov_diag)[None, :]
    return mean[None, :] + z


def rpc_select_nodes(times: torch.Tensor, d: int, num_candidates_per_t: int, num_select: int,
                    proxy_fn: Callable[[torch.Tensor, float], torch.Tensor],
                    mean_fn: Callable[[float], torch.Tensor],
                    cov_diag_fn: Callable[[float], torch.Tensor], device) -> Tuple[torch.Tensor, torch.Tensor]:
    """RPC-like node selection based on proxy variance scores"""
    cand_nodes_t = []
    cand_nodes_x = []
    scores = []
    
    for t in times:
        m = mean_fn(float(t.item()))
        Pdiag = cov_diag_fn(float(t.item()))
        Xcand = sample_prior_states(d, num_candidates_per_t, m, Pdiag, device)
        sc = proxy_fn(Xcand, float(t.item()))  # N
        cand_nodes_t.append(torch.full((num_candidates_per_t, 1), float(t.item()), device=device))
        cand_nodes_x.append(Xcand)
        scores.append(sc)
    
    Tnodes = torch.cat(cand_nodes_t, dim=0)
    Xnodes = torch.cat(cand_nodes_x, dim=0)
    Scores = torch.cat(scores, dim=0)
    
    idx = torch.topk(Scores.flatten(), k=min(num_select, Scores.numel())).indices
    return Tnodes[idx], Xnodes[idx]



def mlmc_two_level_estimator(sde: ItoSDE, x0: torch.Tensor, T: float, dt_coarse: float, dt_fine: float,
                            N_coarse: int, N_fine: int,
                            integrand: Callable[[torch.Tensor], torch.Tensor],
                            seed_base: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-level MLMC estimator with time randomization"""
    device = x0.device
    
    fine_vals = []
    for i in range(N_fine):
        tau = torch.rand(()) * T  # Time randomization
        xT_fine = sde.simulate_final(x0, float(tau.item()), dt_fine, seed_base + i)
        val = integrand(xT_fine) * T  # Scale by T for time randomization
        fine_vals.append(val)
    
    coarse_vals = []
    for i in range(N_coarse):
        tau = torch.rand(()) * T if i >= N_fine else torch.rand(()) * T  # Same tau for coupling
        xT_coarse = sde.simulate_final(x0, float(tau.item()), dt_coarse, seed_base + i)
        val = integrand(xT_coarse) * T
        coarse_vals.append(val)
    
    fine_mean = torch.stack(fine_vals).mean() if fine_vals else torch.tensor(0.0, device=device)
    coarse_mean = torch.stack(coarse_vals).mean() if coarse_vals else torch.tensor(0.0, device=device)
    
    mlmc_est = fine_mean
    
    fine_var = torch.stack(fine_vals).var() if len(fine_vals) > 1 else torch.tensor(0.0, device=device)
    coarse_var = torch.stack(coarse_vals).var() if len(coarse_vals) > 1 else torch.tensor(0.0, device=device)
    
    return mlmc_est, fine_var, coarse_var



def train_critic(critic: CriticNet, sde: ItoSDE, x0: torch.Tensor, T: float, dt: float,
                integrand_fn: Callable, num_epochs: int = 100, batch_size: int = 32,
                lr: float = 1e-3, device=None) -> List[float]:
    """Train critic network using Poisson residual loss"""
    if device is None:
        device = x0.device
    
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    losses = []
    
    for epoch in range(num_epochs):
        batch_t = torch.rand(batch_size, 1, device=device) * T
        batch_x = []
        h_targets = []
        
        for i in range(batch_size):
            t_val = float(batch_t[i].item())
            x_t = sde.simulate_final(x0, t_val, dt, seed=epoch * batch_size + i)
            batch_x.append(x_t)
            
            h_val, _ = integrand_fn(x_t)
            h_targets.append(h_val.unsqueeze(0))
        
        batch_x = torch.stack(batch_x)
        h_targets = torch.stack(h_targets)
        
        optimizer.zero_grad()
        loss = poisson_residual_loss(critic, batch_t, batch_x, h_targets, sde.b)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return losses


if __name__ == "__main__":
    from .preprocess import set_seed, get_device
    
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    d = 5
    r = 2
    
    def b(x):
        return -0.1 * x
    
    def sigma1(x):
        return 0.2 * torch.sin(x)
    
    def sigma2(x):
        return 0.1 * torch.cos(x)
    
    sigmas = [sigma1, sigma2]
    sde = ItoSDE(b, sigmas, r)
    
    x0 = torch.randn(d, device=device)
    T = 1.0
    dt = 0.01
    
    xT = sde.simulate_final(x0, T, dt, seed=42)
    print(f"Initial state norm: {torch.norm(x0).item():.6f}")
    print(f"Final state norm: {torch.norm(xT).item():.6f}")
    
    C = 2  # Number of coefficient heads
    critic = CriticNet(d, C, hidden=64, layers=2).to(device)
    
    t_test = torch.rand(10, 1, device=device)
    x_test = torch.randn(10, d, device=device)
    vals, grads = critic(t_test, x_test)
    print(f"Critic output shapes: vals {vals.shape}, grads[0] {grads[0].shape}")
    
    print("Training module test completed successfully!")
