"""
Models for SEEDS experiments: p_theta and surrogate models
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.seeds_config import config

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    
    def __init__(self, dim: int = 64, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=t.device) / half
        )
        args = t.unsqueeze(-1) * freqs * 2 * math.pi
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)

class ImagePtheta(nn.Module):
    """Denoising model for 2D image-like data"""
    
    def __init__(self, K: int = 16, base_ch: int = 32, time_dim: int = 64):
        super().__init__()
        self.K = K
        self.time_embed = TimeEmbedding(time_dim)
        self.token_emb = nn.Embedding(K, base_ch)
        self.film = nn.Sequential(nn.Linear(time_dim, 2 * base_ch))
        
        self.conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(4, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(4, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(4, base_ch),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(base_ch, K, 1)
    
    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, H, W = xt.shape
        x = self.token_emb(xt)  # [B,H,W,C]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
        
        te = self.time_embed(t.view(B))  # [B,T]
        gamma_beta = self.film(te)  # [B,2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(B, -1, 1, 1)
        beta = beta.view(B, -1, 1, 1)
        
        x = x * (1 + gamma) + beta
        x = self.conv(x)
        logits = self.head(x).permute(0, 2, 3, 1).contiguous()  # [B,H,W,K]
        return logits

class ImageSurrogate(nn.Module):
    """Surrogate model for 2D image-like data"""
    
    def __init__(self, K: int = 16, base_ch: int = 16, time_dim: int = 32):
        super().__init__()
        self.K = K
        self.time_embed = TimeEmbedding(time_dim)
        self.token_emb = nn.Embedding(K, base_ch)
        self.film = nn.Sequential(nn.Linear(time_dim, 2 * base_ch))
        
        self.net = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(),
        )
        self.head_mean = nn.Conv2d(base_ch, 1, 1)
        self.head_logsig = nn.Conv2d(base_ch, 1, 1)
    
    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W = xt.shape
        x = self.token_emb(xt).permute(0, 3, 1, 2)
        
        te = self.time_embed(t.view(B))
        gamma, beta = self.film(te).chunk(2, dim=-1)
        gamma = gamma.view(B, -1, 1, 1)
        beta = beta.view(B, -1, 1, 1)
        
        x = x * (1 + gamma) + beta
        h = self.net(x)
        
        mean = F.softplus(self.head_mean(h)).squeeze(1) + 1e-6  # [B,H,W]
        logsig = self.head_logsig(h).squeeze(1)
        sigma = F.softplus(logsig) + 1e-6
        
        return mean, sigma

class SequencePtheta(nn.Module):
    """Denoising model for 1D sequence data"""
    
    def __init__(self, K: int = 16, d_model: int = 128, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.K = K
        self.emb = nn.Embedding(K, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1024, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.t_emb = TimeEmbedding(d_model)
        self.film = nn.Linear(d_model, 2*d_model)
        self.head = nn.Linear(d_model, K)
    
    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L = xt.shape
        x = self.emb(xt)
        x = x + self.pos_emb[:L].unsqueeze(0)
        
        te = self.t_emb(t.view(B))
        gamma, beta = self.film(te).chunk(2, dim=-1)
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        
        h = self.encoder(x)
        logits = self.head(h)  # [B,L,K]
        return logits

class SequenceSurrogate(nn.Module):
    """Surrogate model for 1D sequence data"""
    
    def __init__(self, K: int = 16, d_model: int = 64, nhead: int = 4, nlayers: int = 1):
        super().__init__()
        self.K = K
        self.emb = nn.Embedding(K, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1024, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.t_emb = TimeEmbedding(d_model)
        self.film = nn.Linear(d_model, 2*d_model)
        self.head_mean = nn.Linear(d_model, 1)
        self.head_logsig = nn.Linear(d_model, 1)
    
    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = xt.shape
        x = self.emb(xt)
        x = x + self.pos_emb[:L].unsqueeze(0)
        
        te = self.t_emb(t.view(B))
        gamma, beta = self.film(te).chunk(2, dim=-1)
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        
        h = self.encoder(x)
        
        mean = F.softplus(self.head_mean(h)).squeeze(-1) + 1e-6  # [B,L]
        sigma = F.softplus(self.head_logsig(h)).squeeze(-1) + 1e-6
        
        return mean, sigma
