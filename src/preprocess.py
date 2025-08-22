
import math
import time
import random
from typing import Callable, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """Get available device (CUDA if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def stratonovich_drift(b: Callable, sigmas: List[Callable], x: torch.Tensor) -> torch.Tensor:
    """Convert Itô drift to Stratonovich drift using JVP-only approach
    b^∘(x) = b(x) - 0.5 sum_j J(sigma_j)(x) sigma_j(x)
    """
    corr = torch.zeros_like(x)
    for sj in sigmas:
        v = sj(x)
        _, jvp_val = jvp(sj, (x,), (v,), create_graph=False, strict=True)
        corr = corr + 0.5 * jvp_val
    return b(x) - corr


def lie_bracket(V: Callable, W: Callable, x: torch.Tensor) -> torch.Tensor:
    """Compute Lie bracket [V, W](x) = J_W(x) V(x) - J_V(x) W(x) using JVPs"""
    Vx = V(x)
    Wx = W(x)
    _, JW_Vx = jvp(W, (x,), (Vx,), create_graph=False, strict=True)
    _, JV_Wx = jvp(V, (x,), (Wx,), create_graph=False, strict=True)
    return JW_Vx - JV_Wx


def nested_bracket(V: Callable, W: Callable, U: Callable, x: torch.Tensor) -> torch.Tensor:
    """Compute nested bracket [V, [W, U]](x)"""
    def inner(z):
        return lie_bracket(W, U, z)
    return lie_bracket(V, inner, x)



def estimate_diffusion_subspace(sigmas: List[Callable], x: torch.Tensor, q: int = 16, 
                                power_iter: int = 1, energy_thresh: float = 0.99) -> torch.Tensor:
    """Estimate local active diffusion subspace using randomized range finding"""
    S = torch.stack([sj(x) for sj in sigmas], dim=1)  # d × r
    d, r_full = S.shape
    if r_full == 0:
        return torch.eye(d, device=x.device, dtype=x.dtype)
    
    q_eff = min(q, r_full)
    Omega = torch.randn(r_full, q_eff, device=x.device, dtype=x.dtype)
    Y = S @ Omega
    for _ in range(power_iter):
        Y = S @ (S.T @ Y)
    Q, _ = torch.linalg.qr(Y, mode='reduced')  # d × q_eff
    B = Q.T @ S
    
    U_s, Sv, _ = torch.linalg.svd(B, full_matrices=False)
    if torch.sum(Sv**2) <= 1e-12:
        return Q @ U_s  # arbitrary
    
    energy = (Sv**2).cumsum(0) / (Sv**2).sum()
    r_hat = int(torch.searchsorted(energy, torch.tensor(energy_thresh, device=energy.device)).item() + 1)
    Ur = (Q @ U_s[:, :r_hat])  # d × r_hat
    return Ur


@torch.no_grad()
def oja_update(probes: torch.Tensor, op_apply: Callable[[torch.Tensor], torch.Tensor], 
               lr: float = 0.1, iters: int = 2) -> torch.Tensor:
    """Oja-style probe alignment for better bracket contraction estimation"""
    Q, _ = torch.linalg.qr(probes, mode='reduced')
    P = Q.clone()
    for _ in range(iters):
        Y = op_apply(P)
        P = P + lr * Y
        P, _ = torch.linalg.qr(P, mode='reduced')
    return P



def hutchinson_norm_sq(vec: torch.Tensor, m: int = 8) -> torch.Tensor:
    """Estimates ||vec||^2 via E[(z·vec)^2], z~N(0,I): unbiased for Euclidean norm"""
    d = vec.numel()
    Z = torch.randn(d, m, device=vec.device, dtype=vec.dtype)
    proj = (Z * vec[:, None]).sum(dim=0)
    return (proj.pow(2).mean())


def hutchpp_trace_symmetric(op_mul: Callable[[torch.Tensor], torch.Tensor], d: int, 
                           m: int = 8, Q_init: torch.Tensor = None) -> torch.Tensor:
    """Hutch++ trace estimator for symmetric operators"""
    device = Q_init.device if Q_init is not None else torch.device('cpu')
    G = torch.randn(d, m, device=device)
    Y = op_mul(G)
    Q, _ = torch.linalg.qr(Y, mode='reduced')
    if Q_init is not None:
        Q = torch.linalg.qr(torch.cat([Q, Q_init], dim=1), mode='reduced')[0]
    B = Q.T @ op_mul(Q)
    Z = torch.randn(d, m, device=device, dtype=Q.dtype)
    PZ = Z - Q @ (Q.T @ Z)
    R = op_mul(PZ)
    trace_resid = torch.sum(PZ * R) / m
    return torch.trace(B) + trace_resid



def make_quadratic_Q(d: int, bandwidth: int = 3, scale: float = 0.5, seed: int = 0) -> torch.Tensor:
    """Create a banded quadratic form matrix for testing"""
    torch.manual_seed(seed)
    Q = torch.zeros(d, d)
    for i in range(d):
        for j in range(max(0, i - bandwidth), min(d, i + bandwidth + 1)):
            Q[i, j] = torch.randn(()) * scale / (1 + abs(i - j))
    Q = (Q + Q.T) / 2.0
    return Q


def g_quadratic(x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Quadratic observable g(x) = x^T Q x"""
    return (x * (Q @ x)).sum()


def grad_g_quadratic(x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Gradient of quadratic observable"""
    return (Q + Q.T) @ x


def g_cubic(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Cubic observable g(x) = sum w_i x_i^3"""
    return (w * x.pow(3)).sum()


def grad_g_cubic(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Gradient of cubic observable"""
    return 3.0 * w * x.pow(2)



def evolve_gaussian_moments(m0: torch.Tensor, P0: torch.Tensor, b: Callable, 
                           sigmas: List[Callable], T: float, dt: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evolve Gaussian moments using linearized dynamics for occupancy prior"""
    m = m0.clone()
    P = P0.clone()
    d = m0.numel()
    
    steps = max(1, int(T / dt))
    for _ in range(steps):
        drift_m = b(m)
        
        eps = 1e-6
        J = torch.zeros(d, d, device=m.device, dtype=m.dtype)
        for i in range(d):
            m_plus = m.clone()
            m_plus[i] += eps
            m_minus = m.clone()
            m_minus[i] -= eps
            J[:, i] = (b(m_plus) - b(m_minus)) / (2 * eps)
        
        if len(sigmas) > 0:
            S = torch.stack([sj(m) for sj in sigmas], dim=1)  # d × r
            Q_diff = S @ S.T
        else:
            Q_diff = torch.zeros_like(P)
        
        m = m + drift_m * dt
        P = P + (J @ P + P @ J.T + Q_diff) * dt
        
        P = P + torch.eye(d, device=P.device, dtype=P.dtype) * 1e-8
    
    return m, P


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    d = 10
    r = 3
    x = torch.randn(d, device=device)
    
    def sigma1(x):
        return torch.sin(x)
    def sigma2(x):
        return torch.cos(x)
    def sigma3(x):
        return x * 0.5
    
    sigmas = [sigma1, sigma2, sigma3]
    
    Ur = estimate_diffusion_subspace(sigmas, x, q=8)
    print(f"Estimated diffusion subspace dimension: {Ur.shape[1]}")
    
    def b(x):
        return -0.1 * x
    
    bracket = lie_bracket(b, sigma1, x)
    print(f"Lie bracket norm: {torch.norm(bracket).item():.6f}")
    
    vec = torch.randn(d, device=device)
    true_norm_sq = torch.norm(vec).pow(2)
    est_norm_sq = hutchinson_norm_sq(vec, m=16)
    print(f"True norm^2: {true_norm_sq.item():.6f}, Estimated: {est_norm_sq.item():.6f}")
    
    print("Preprocessing module test completed successfully!")
