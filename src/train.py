"""
GASP Model Implementation for Latency Benchmarking

This module implements a mock GASP (Grouped Autoregressive Scale Prediction) model
that simulates the computational characteristics of the proposed architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupConditioningAdapter(nn.Module):
    """
    Group Conditioning Adapter (GCA) - A lightweight cross-attention module
    that simulates the computational overhead of conditioning on group history.
    """
    def __init__(self, embed_dim=1024, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, history_features, group_size):
        """
        Simulate cross-attention computation that scales with group size.
        
        Args:
            history_features: Previous generation history
            group_size: Current group size being generated
            
        Returns:
            Conditioned features
        """
        batch_size = history_features.shape[0]
        
        group_queries = torch.randn(
            batch_size, group_size * 4, self.embed_dim,  # 4 tokens per scale
            device=history_features.device,
            dtype=history_features.dtype
        )
        
        Q = self.query_proj(group_queries)
        K = self.key_proj(history_features)
        V = self.value_proj(history_features)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(Q, K, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        output = self.out_proj(attn_output)
        
        return self.norm(output + group_queries)

class HierarchicalGroupCausalMask(nn.Module):
    """
    Simulates the hierarchical group-causal attention mask computation.
    This represents the overhead of managing causality between groups
    while allowing parallel attention within groups.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.mask_generator = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, features, group_size):
        """
        Simulate mask computation overhead that scales with group size.
        """
        mask_features = self.mask_generator(features)
        
        group_tokens = torch.randn(
            features.shape[0], group_size, features.shape[-1],
            device=features.device, dtype=features.dtype
        )
        
        _ = torch.bmm(group_tokens, group_tokens.transpose(1, 2))
        
        return mask_features

class GASPModel(nn.Module):
    """
    Mock GASP model that simulates the computational characteristics
    of Grouped Autoregressive Scale Prediction.
    """
    def __init__(self, embed_dim=1024, num_layers=16, num_heads=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        self.gca = GroupConditioningAdapter(embed_dim, num_heads)
        self.mask_module = HierarchicalGroupCausalMask(embed_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.0,  # No dropout for consistent timing
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for realistic computation."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def generate(self, num_tokens, group_size):
        """
        Simulates the Group-wise Ancestral Sampling loop.
        
        Args:
            num_tokens: Total number of tokens to generate
            group_size: Number of scales to generate per step
            
        Returns:
            Generated features (for timing purposes)
        """
        if num_tokens % group_size != 0:
            raise ValueError(f"num_tokens ({num_tokens}) must be divisible by group_size ({group_size})")
        
        num_steps = num_tokens // group_size
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        history_features = torch.randn(1, 1, self.embed_dim, device=device, dtype=dtype)
        
        for step in range(num_steps):
            conditioned_features = self.gca(history_features, group_size)
            
            masked_features = self.mask_module(conditioned_features, group_size)
            
            current_features = masked_features.mean(dim=1, keepdim=True)  # Aggregate to single token
            
            for layer in self.transformer_layers:
                current_features = layer(current_features)
            
            output_features = self.output_proj(current_features)
            
            history_features = torch.cat([history_features, output_features], dim=1)
            
            if history_features.shape[1] > 100:
                history_features = history_features[:, -50:]  # Keep last 50 tokens

        return history_features

    def get_model_info(self):
        """Return model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
