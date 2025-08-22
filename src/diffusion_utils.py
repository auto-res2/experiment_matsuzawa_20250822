"""
Diffusion utilities: beta schedule, forward process, coupling factors
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.seeds_config import BetaParams

class BetaSchedule:
    """Beta schedule for discrete diffusion"""
    
    def __init__(self, K: int, params: BetaParams = BetaParams()):
        self.beta0 = params.beta0
        self.beta1 = params.beta1
        self.gamma = params.gamma
        self.K = K
        self.F1 = self._F(torch.tensor(1.0))
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta0 + (self.beta1 - self.beta0) * t.pow(self.gamma)
    
    def _F(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta0 * t + (self.beta1 - self.beta0) * t.pow(self.gamma + 1) / (self.gamma + 1.0)
    
    def V_of_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.F1 - self._F(t)
    
    def t_of_V(self, V: torch.Tensor, iters: int = 8, eps: float = 1e-10) -> torch.Tensor:
        V = V.clamp(min=0.0, max=float(self.F1))
        target = self.F1 - V
        t = (target / self.F1).clamp(0.0, 1.0)
        for _ in range(iters):
            Ft = self._F(t)
            bt = self.beta(t)
            step = (Ft - target) / (bt + eps)
            t = (t - step).clamp(0.0, 1.0)
        return t
    
    def pi(self, t: torch.Tensor) -> torch.Tensor:
        Ft = self._F(t)
        alpha = torch.exp(-Ft)
        return alpha + (1.0 - alpha) / float(self.K)

def forward_corrupt(x0: torch.Tensor, t: torch.Tensor, K: int, sched: BetaSchedule) -> torch.Tensor:
    """
    Forward corruption process
    x0: [...,] integer tensor in [0,K-1]
    t: scalar tensor or tensor broadcastable to x0
    returns xt: same shape integer tensor
    """
    device = x0.device
    pi = sched.pi(t).to(device)
    
    while pi.dim() < x0.dim():
        pi = pi.unsqueeze(-1)
    
    keep = torch.bernoulli(pi.expand_as(x0).float()).bool()
    xt = x0.clone()
    
    rand = torch.randint(low=0, high=K - 1, size=x0.shape, device=device)
    mapped = rand + (rand >= x0).long()
    xt[~keep] = mapped[~keep]
    
    return xt

def E_site_from_logits(logits: torch.Tensor, xi: torch.Tensor, t: torch.Tensor, 
                      sched: BetaSchedule, K: int) -> torch.Tensor:
    """Compute site-level coupling factor E_site"""
    p = F.softmax(logits, dim=-1)
    pi = sched.pi(t)
    
    while pi.dim() < p.dim() - 1:
        pi = pi.unsqueeze(-1)
    
    r1 = (1.0 - pi) / ((K - 1) * pi)
    r2 = (K - 1) * pi / (1.0 - pi)
    
    p_xi = p.gather(-1, xi.unsqueeze(-1)).squeeze(-1)
    term = 1.0 + p_xi * (r1 - 1.0) + ((1.0 - p_xi) / (K - 1)) * (r2 - 1.0)
    
    return term

def E_pair_from_logits(logits: torch.Tensor, xi: torch.Tensor, t: torch.Tensor, 
                      sched: BetaSchedule, K: int) -> torch.Tensor:
    """Compute pair-level coupling factor E_pair"""
    p = F.softmax(logits, dim=-1)
    pi = sched.pi(t)
    
    while pi.dim() < p.dim() - 1:
        pi = pi.unsqueeze(-1)
    
    r1 = (1.0 - pi) / ((K - 1) * pi)
    r2 = (K - 1) * pi / (1.0 - pi)
    
    p_xi = p.gather(-1, xi.unsqueeze(-1))  # [...,1]
    E_pair = 1.0 + p_xi * (r1 - 1.0) + p * (r2 - 1.0)
    
    one_hot = F.one_hot(xi.long(), num_classes=K).bool()
    E_pair = E_pair.masked_fill(one_hot, 0.0)
    
    return E_pair

def heteroscedastic_loss(pred_mean: torch.Tensor, pred_sigma: torch.Tensor, 
                        target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic loss for surrogate training"""
    diff2 = (target - pred_mean).pow(2)
    loss = 0.5 * diff2 / (pred_sigma.pow(2) + 1e-8) + torch.log(pred_sigma + 1e-8)
    return loss.mean()
