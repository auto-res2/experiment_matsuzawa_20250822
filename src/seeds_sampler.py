"""
SEEDS sampler implementation
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import time
from dataclasses import dataclass
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.seeds_config import config
from src.diffusion_utils import BetaSchedule, E_site_from_logits, E_pair_from_logits

@dataclass
class SEEDSStats:
    """Statistics for SEEDS sampling"""
    nfe_heavy: int = 0  # Number of p_theta evaluations
    nfe_surrogate: int = 0  # Number of surrogate evaluations
    n_events: int = 0  # Number of accepted events
    n_candidates: int = 0  # Number of candidate events
    wall_time: float = 0.0
    acceptance_rate: float = 0.0
    bound_violations: int = 0

class SEEDSSampler:
    """SEEDS event-driven sampler"""
    
    def __init__(self, ptheta_model, surrogate_model, kappa: float, K: int, 
                 sched: BetaSchedule, device: str = 'cpu', exact_mode: bool = True,
                 p_corr: float = 0.8):
        self.ptheta = ptheta_model
        self.surrogate = surrogate_model
        self.kappa = kappa
        self.K = K
        self.sched = sched
        self.device = device
        self.exact_mode = exact_mode
        self.p_corr = p_corr
        
        self.ptheta.to(device)
        self.surrogate.to(device)
        self.ptheta.eval()
        self.surrogate.eval()
    
    def sample_tau_leaping_baseline(self, x_init: torch.Tensor, n_steps: int = 100) -> Tuple[torch.Tensor, SEEDSStats]:
        """Baseline tau-leaping sampler for comparison"""
        stats = SEEDSStats()
        start_time = time.time()
        
        xt = x_init.clone()
        dt = 1.0 / n_steps
        
        with torch.no_grad():
            for step in range(n_steps):
                t = torch.tensor(1.0 - step * dt, device=self.device)
                
                logits = self.ptheta(xt.unsqueeze(0), t.unsqueeze(0)).squeeze(0)
                stats.nfe_heavy += 1
                
                beta_t = self.sched.beta(t)
                E_pair = E_pair_from_logits(logits.unsqueeze(0), xt.unsqueeze(0), t, self.sched, self.K).squeeze(0)
                
                for i in range(xt.numel()):
                    flat_idx = i
                    site_coords = np.unravel_index(flat_idx, xt.shape)
                    
                    xi = xt[site_coords].item()
                    
                    rates = beta_t * E_pair[site_coords] / (self.K - 1)  # Uniform base rates
                    total_rate = rates.sum().item()
                    
                    n_jumps = torch.poisson(torch.tensor(total_rate * dt)).int().item()
                    
                    if n_jumps > 0:
                        probs = rates / (total_rate + 1e-8)
                        probs[xi] = 0.0
                        probs = probs / (probs.sum() + 1e-8)
                        
                        if probs.sum() > 1e-8:
                            new_state = torch.multinomial(probs, 1).item()
                            xt[site_coords] = new_state
                            stats.n_events += 1
        
        stats.wall_time = time.time() - start_time
        return xt, stats
    
    def sample_seeds(self, x_init: torch.Tensor, max_events: int = 1000) -> Tuple[torch.Tensor, SEEDSStats]:
        """SEEDS event-driven sampling"""
        stats = SEEDSStats()
        start_time = time.time()
        
        xt = x_init.clone()
        t_current = 1.0  # Start from t=1
        
        with torch.no_grad():
            for event_idx in range(max_events):
                if t_current <= 1e-6:  # Close to t=0
                    break
                
                t_tensor = torch.tensor(t_current, device=self.device)
                mean, sigma = self.surrogate(xt.unsqueeze(0), t_tensor.unsqueeze(0))
                mean = mean.squeeze(0)
                sigma = sigma.squeeze(0)
                stats.nfe_surrogate += 1
                
                rho = mean + self.kappa * sigma
                
                beta_t = self.sched.beta(t_tensor)
                Lambda_bar = beta_t * rho  # [H, W] or [L]
                
                total_rate = Lambda_bar.sum().item()
                
                if total_rate < 1e-8:
                    break
                
                dt = torch.distributions.Exponential(total_rate).sample().item()
                t_next = max(0.0, t_current - dt)
                
                if t_next <= 1e-6:
                    break
                
                site_probs = Lambda_bar.flatten() / total_rate
                site_idx = torch.multinomial(site_probs, 1).item()
                site_coords = np.unravel_index(site_idx, xt.shape)
                
                stats.n_candidates += 1
                
                xi = xt[site_coords].item()
                
                if self.exact_mode:
                    t_tensor = torch.tensor(t_next, device=self.device)
                    logits = self.ptheta(xt.unsqueeze(0), t_tensor.unsqueeze(0)).squeeze(0)
                    stats.nfe_heavy += 1
                    
                    E_site_exact = E_site_from_logits(
                        logits.unsqueeze(0), xt.unsqueeze(0), t_tensor, self.sched, self.K
                    ).squeeze(0)
                    
                    E_site_val = E_site_exact[site_coords].item()
                    rho_val = rho[site_coords].item()
                    
                    accept_prob = E_site_val / (rho_val + 1e-8)
                    
                    if rho_val < E_site_val:
                        stats.bound_violations += 1
                        accept_prob = 1.0  # Accept anyway but count violation
                else:
                    E_site_val = mean[site_coords].item()
                    rho_val = rho[site_coords].item()
                    accept_prob = E_site_val / (rho_val + 1e-8)
                    
                    if torch.rand(1).item() < self.p_corr:
                        t_tensor = torch.tensor(t_next, device=self.device)
                        logits = self.ptheta(xt.unsqueeze(0), t_tensor.unsqueeze(0)).squeeze(0)
                        stats.nfe_heavy += 1
                        
                        E_site_exact = E_site_from_logits(
                            logits.unsqueeze(0), xt.unsqueeze(0), t_tensor, self.sched, self.K
                        ).squeeze(0)
                        
                        E_site_val = E_site_exact[site_coords].item()
                        accept_prob = E_site_val / (rho_val + 1e-8)
                
                if torch.rand(1).item() < accept_prob:
                    alternatives = list(range(self.K))
                    alternatives.remove(xi)
                    new_state = np.random.choice(alternatives)
                    xt[site_coords] = new_state
                    
                    stats.n_events += 1
                
                t_current = t_next
        
        stats.wall_time = time.time() - start_time
        stats.acceptance_rate = stats.n_events / max(1, stats.n_candidates)
        
        return xt, stats

def compare_samplers(ptheta_model, surrogate_model, kappa: float, x_init: torch.Tensor,
                    K: int, sched: BetaSchedule, device: str = 'cpu') -> Dict[str, Any]:
    """Compare SEEDS vs tau-leaping baseline"""
    
    sampler = SEEDSSampler(ptheta_model, surrogate_model, kappa, K, sched, device)
    
    x_seeds_exact, stats_seeds_exact = sampler.sample_seeds(x_init.clone(), max_events=500)
    
    sampler.exact_mode = False
    x_seeds_budgeted, stats_seeds_budgeted = sampler.sample_seeds(x_init.clone(), max_events=500)
    
    x_tau, stats_tau = sampler.sample_tau_leaping_baseline(x_init.clone(), n_steps=50)
    
    return {
        'seeds_exact': {'sample': x_seeds_exact, 'stats': stats_seeds_exact},
        'seeds_budgeted': {'sample': x_seeds_budgeted, 'stats': stats_seeds_budgeted},
        'tau_leaping': {'sample': x_tau, 'stats': stats_tau}
    }
