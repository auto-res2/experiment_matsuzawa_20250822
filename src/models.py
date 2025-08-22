#!/usr/bin/env python

"""
Core model implementations for CAMoE-Diff experiment.
Contains the Content-Aware Mixture-of-Experts Diffusion model and baseline models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fvcore.nn import FlopCountAnalysis
from typing import Dict, List, Tuple, Optional


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd // 4, n_embd)
        self.linear_2 = nn.Linear(n_embd, n_embd)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.linear_1.in_features // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.linear_2(F.silu(self.linear_1(emb)))


class IdentityExpert(nn.Module):
    """Identity expert - near-zero computational cost."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConvExpert(nn.Module):
    """Convolutional expert for local texture processing."""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x))) + x  # Residual connection


class DilatedConvExpert(nn.Module):
    """Dilated convolutional expert for larger receptive field."""
    
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x))) + x  # Residual connection


class AttentionExpert(nn.Module):
    """Self-attention expert for global context modeling."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        x_norm = self.norm(x)
        x_flat = x_norm.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        return attn_out + x


class GatingNetwork(nn.Module):
    """Content-aware gating network for expert selection."""
    
    def __init__(self, channels: int, num_experts: int, use_content: bool = True):
        super().__init__()
        self.use_content = use_content
        self.num_experts = num_experts
        
        self.time_emb = nn.Linear(channels, channels)
        
        if self.use_content:
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(channels, channels // 2, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(channels // 2, num_experts, 1)
            )
        else:
            self.global_fc = nn.Linear(channels, num_experts)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
            t_emb: Time embedding (B, C)
        Returns:
            Routing logits (B, num_experts, H, W)
        """
        t_proj = self.time_emb(F.silu(t_emb))  # (B, C)
        
        if self.use_content:
            x_with_time = x + t_proj[:, :, None, None]
            logits = self.spatial_conv(x_with_time)  # (B, num_experts, H, W)
        else:
            logits = self.global_fc(t_proj)  # (B, num_experts)
            logits = logits.unsqueeze(-1).unsqueeze(-1)  # (B, num_experts, 1, 1)
            logits = logits.expand(-1, -1, x.shape[2], x.shape[3])  # (B, num_experts, H, W)
        
        return logits


class MoEBlock(nn.Module):
    """Mixture-of-Experts block with content-aware routing."""
    
    def __init__(self, channels: int, num_experts: int = 4, use_content: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.use_content = use_content
        
        self.gating_network = GatingNetwork(channels, num_experts, use_content)
        
        self.experts = nn.ModuleList([
            IdentityExpert(),                           # Expert 0: Identity (cheapest)
            ConvExpert(channels, kernel_size=3),        # Expert 1: 3x3 Conv
            DilatedConvExpert(channels, kernel_size=7), # Expert 2: 7x7 Dilated Conv
            AttentionExpert(channels)                   # Expert 3: Self-Attention (most expensive)
        ])
        
        self.register_buffer('expert_costs', self._compute_expert_costs(channels))
        
    def _compute_expert_costs(self, channels: int) -> torch.Tensor:
        """Pre-computes FLOPs for each expert."""
        costs = []
        dummy_input = torch.randn(1, channels, 32, 32)  # Use smaller size for FLOP counting
        
        for expert in self.experts:
            try:
                flops = FlopCountAnalysis(expert, dummy_input).total()
                costs.append(float(flops))
            except Exception:
                if isinstance(expert, IdentityExpert):
                    costs.append(0.0)
                elif isinstance(expert, ConvExpert):
                    costs.append(1e6)
                elif isinstance(expert, DilatedConvExpert):
                    costs.append(2e6)
                elif isinstance(expert, AttentionExpert):
                    costs.append(5e6)
                else:
                    costs.append(1e6)
        
        return torch.tensor(costs, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            x: Input features (B, C, H, W)
            t_emb: Time embedding (B, C)
        Returns:
            output: Mixed expert outputs (B, C, H, W)
            aux_losses: Dictionary with 'cost' and 'balance' losses
            routing_decisions: Hard routing decisions (B, H, W)
        """
        logits = self.gating_network(x, t_emb)  # (B, num_experts, H, W)
        
        soft_weights = F.softmax(logits, dim=1)  # (B, num_experts, H, W)
        
        hard_weights = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)  # (B, num_experts, H, W)
        
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, C, H, W)
        
        output = torch.einsum('bnchw,bnhw->bchw', expert_outputs, hard_weights)
        
        pixel_costs = torch.einsum('bnhw,n->bhw', soft_weights, self.expert_costs.to(x.device))
        cost_loss = pixel_costs.mean()
        
        expert_usage = soft_weights.mean(dim=(0, 2, 3))  # Average usage per expert
        balance_loss = expert_usage.var() / (expert_usage.mean() + 1e-8)
        
        aux_losses = {
            'cost': cost_loss,
            'balance': balance_loss
        }
        
        routing_decisions = hard_weights.argmax(dim=1)  # (B, H, W)
        
        return output, aux_losses, routing_decisions


class BaseUNet(nn.Module):
    """Base U-Net architecture for diffusion models."""
    
    def __init__(self, model_type: str = 'ADM', config: Optional[Dict] = None):
        super().__init__()
        self.model_type = model_type
        
        if config is None:
            config = {
                'image_size': 64,
                'in_channels': 3,
                'base_channels': 64,
                'num_experts': 4
            }
        
        self.config = config
        C = config['base_channels']
        
        self.time_embedding = TimeEmbedding(C)
        
        self.initial_conv = nn.Conv2d(config['in_channels'], C, 3, padding=1)
        
        if model_type == 'CAMoE-Diff':
            self.blocks = nn.ModuleList([
                MoEBlock(C, num_experts=config['num_experts'], use_content=True),
                MoEBlock(C, num_experts=config['num_experts'], use_content=True)
            ])
        elif model_type == 'Content-Agnostic':
            self.blocks = nn.ModuleList([
                MoEBlock(C, num_experts=config['num_experts'], use_content=False),
                MoEBlock(C, num_experts=config['num_experts'], use_content=False)
            ])
        elif model_type == 'ADM':
            self.blocks = nn.ModuleList([
                AttentionExpert(C),
                AttentionExpert(C)
            ])
        elif model_type == 'PCDM':
            self.blocks = nn.ModuleList([
                ConvExpert(C),  # Cheaper for early stages
                AttentionExpert(C)  # More expensive for later stages
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.final_conv = nn.Conv2d(C, config['in_channels'], 1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: Noisy input images (B, C, H, W)
            t: Timesteps (B,)
        Returns:
            predicted_noise: Predicted noise (B, C, H, W)
            aux_losses: Auxiliary losses for MoE models
            routing_decisions: List of routing decisions for each MoE block
        """
        t_emb = self.time_embedding(t)
        
        h = self.initial_conv(x)
        
        aux_losses = {'cost': torch.tensor(0.0, device=x.device), 'balance': torch.tensor(0.0, device=x.device)}
        routing_decisions = []
        
        for block in self.blocks:
            if isinstance(block, MoEBlock):
                h, block_aux_losses, routing = block(h, t_emb)
                aux_losses['cost'] += block_aux_losses['cost']
                aux_losses['balance'] += block_aux_losses['balance']
                routing_decisions.append(routing)
            else:
                h = block(h)
        
        predicted_noise = self.final_conv(h)
        
        return predicted_noise, aux_losses, routing_decisions


def create_model(model_type: str, config: Dict) -> BaseUNet:
    """Factory function to create models."""
    return BaseUNet(model_type=model_type, config=config)


if __name__ == "__main__":
    config = {
        'image_size': 64,
        'in_channels': 3,
        'base_channels': 64,
        'num_experts': 4
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_types = ['CAMoE-Diff', 'Content-Agnostic', 'ADM', 'PCDM']
    
    for model_type in model_types:
        print(f"\nTesting {model_type} model...")
        model = create_model(model_type, config).to(device)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64, device=device)
        t = torch.randint(0, 1000, (batch_size,), device=device).float()
        
        with torch.no_grad():
            pred_noise, aux_losses, routing = model(x, t)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {pred_noise.shape}")
        print(f"  Cost loss: {aux_losses['cost'].item():.6f}")
        print(f"  Balance loss: {aux_losses['balance'].item():.6f}")
        print(f"  Routing decisions: {len(routing)} blocks")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    print("\nModel testing completed successfully!")
