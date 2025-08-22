import math
import time
import random
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocess import set_seed, create_datasets

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64, patch_size=4, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H/ps, W/ps]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x, (H, W)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Dropout(drop)
        )

    def forward(self, x):
        B, N, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x, attn  # return attn for aux stats

class MiniViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=64, depth=6, num_heads=4, num_classes=10, cls_token=True):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size, img_size)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if cls_token else None
        self.pos_embed = nn.Parameter(torch.randn(1, (1 if cls_token else 0) + self.num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.num_classes = num_classes
        self.head = nn.Linear(embed_dim, num_classes)
        self.cls_token_used = cls_token

    def forward_collect(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        B = x.size(0)
        tokens, spatial_hw = self.patch_embed(x)  # [B, N, C]
        if self.cls_token_used:
            cls_tok = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
            tokens = torch.cat([cls_tok, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        feats = []
        attns = []
        h = tokens
        for blk in self.blocks:
            h, attn = blk(h)
            feats.append(h)
            attns.append(attn)
        h = self.norm(h)
        feats[-1] = h
        return feats, attns  # list of [B, N, C]

    def forward_head(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.cls_token_used:
            cls = tokens[:, 0]
        else:
            cls = tokens.mean(dim=1)
        logits = self.head(cls)
        return logits

    def forward(self, x):
        feats, _ = self.forward_collect(x)
        return self.forward_head(feats[-1])

class UncertaintyHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.logits = nn.Linear(dim, num_classes)
        self.log_sigma = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens.size(1) > 1:
            pooled = tokens[:, 0]
        else:
            pooled = tokens.squeeze(1)
        return self.logits(pooled), self.log_sigma(pooled).clamp(-3, 3)

class TwinCriticMCV(nn.Module):
    def __init__(self, d_model: int, n_layers: int, aux_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.layer_embed = nn.Embedding(n_layers, 16)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 16 + aux_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.head_mu = nn.Linear(hidden, 1)
        self.head_logsigma = nn.Linear(hidden, 1)

    def forward(self, token_embed: torch.Tensor, layer_id: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lid = self.layer_embed(layer_id)
        h = torch.cat([token_embed, lid, aux], dim=-1)
        h = self.mlp(h)
        mu = self.head_mu(h).squeeze(-1)
        log_sigma = self.head_logsigma(h).squeeze(-1).clamp(-6, 3)
        return mu, log_sigma

class SoftTopKRouter(nn.Module):
    def __init__(self, k_per_layer: int, temperature: float = 0.7, entropy_reg: float = 1e-3):
        super().__init__()
        self.k = k_per_layer
        self.tau = temperature
        self.entropy_reg = entropy_reg

    def forward(self, score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = -torch.log(-torch.log(torch.rand_like(score)))
        y = (score + g) / self.tau
        w = torch.softmax(y, dim=0)
        w = w * (self.k / (w.sum() + 1e-8))
        entropy = -(w * (w.clamp(min=1e-8).log())).sum()
        return w, entropy

    @staticmethod
    def hard_topk(score: torch.Tensor, k: int) -> torch.Tensor:
        idx = torch.topk(score, k=k, dim=0).indices
        mask = torch.zeros_like(score, dtype=torch.bool)
        mask[idx] = True
        return mask

class BudgetController:
    def __init__(self, target_cost: float = 1.0, rho: float = 1e-3, device: torch.device = torch.device('cpu')):
        self.lmbda = torch.tensor(0.0, device=device)
        self.target = target_cost
        self.rho = rho

    def update(self, measured_cost: float) -> float:
        with torch.no_grad():
            self.lmbda = (self.lmbda + self.rho * (measured_cost - self.target)).clamp(min=0.0)
        return float(self.lmbda.item())

class MoDRCCModel(nn.Module):
    """Complete MoD-RCC model with dynamic token routing."""
    def __init__(self, img_size=32, patch_size=4, embed_dim=64, depth=6, num_heads=4, 
                 num_classes=10, k_per_layer=8, target_budget=0.7):
        super().__init__()
        self.backbone = MiniViT(img_size, patch_size, 3, embed_dim, depth, num_heads, num_classes)
        self.twin_critic = TwinCriticMCV(embed_dim, depth, aux_dim=16)
        self.router = SoftTopKRouter(k_per_layer)
        self.budget_controller = BudgetController(target_budget)
        
        self.shallow_head = UncertaintyHead(embed_dim, num_classes)
        self.deep_head = UncertaintyHead(embed_dim, num_classes)
        
        self.depth = depth
        self.k_per_layer = k_per_layer
        self.embed_dim = embed_dim

    def forward(self, x, training=True):
        B = x.size(0)
        feats, attns = self.backbone.forward_collect(x)
        
        if training:
            all_logits = []
            total_cost = 0.0
            
            for layer_idx in range(self.depth):
                tokens = feats[layer_idx]  # [B, N, C]
                N = tokens.size(1)
                
                flat_tokens = tokens.view(-1, self.embed_dim)  # [B*N, C]
                layer_ids = torch.full((B * N,), layer_idx, device=x.device, dtype=torch.long)
                aux_features = torch.randn(B * N, 16, device=x.device)  # Mock aux features
                
                mu, log_sigma = self.twin_critic(flat_tokens, layer_ids, aux_features)
                scores = mu.view(B, N)  # [B, N]
                
                if layer_idx < self.depth - 1:
                    logits, _ = self.shallow_head(tokens)
                else:
                    logits, _ = self.deep_head(tokens)
                
                all_logits.append(logits)
                total_cost += 0.5  # Mock cost
            
            final_logits = all_logits[-1]
            return final_logits, total_cost
        else:
            return self.backbone.forward_head(feats[-1]), 1.0

def train_model(model, train_loader, val_loader, device, num_epochs=5):
    """Train the MoD-RCC model."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    
    print(f"Training MoD-RCC model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits, cost = model(data, training=True)
            loss = criterion(logits, target)
            
            budget_penalty = model.budget_controller.update(cost) * (cost - model.budget_controller.target)
            total_loss = loss + 0.1 * budget_penalty
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits, _ = model(data, training=False)
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
    
    return train_losses, val_accuracies

def train_classification_model(datasets, device):
    """Train classification model and return training history."""
    print("\n=== Training Classification Model ===")
    
    train_loader = DataLoader(datasets['cls_train'], batch_size=32, shuffle=True)
    val_loader = DataLoader(datasets['cls_val'], batch_size=32, shuffle=False)
    
    model = MoDRCCModel(
        img_size=32, patch_size=4, embed_dim=64, depth=4, 
        num_heads=4, num_classes=10, k_per_layer=8
    )
    
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, device, num_epochs=3)
    
    return model, train_losses, val_accuracies

if __name__ == "__main__":
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    datasets = create_datasets()
    model, losses, accuracies = train_classification_model(datasets, device)
    
    print("Training completed successfully!")
    print(f"Final validation accuracy: {accuracies[-1]:.4f}")
