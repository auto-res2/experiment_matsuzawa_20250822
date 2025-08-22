#!/usr/bin/env python3
"""
Training script for DASH-HiLo-Anchor SHViT models.
Implements the core model architecture and training loop.
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class ChannelScorer(nn.Module):
    """Channel scoring module for DASH channel co-budgeting."""
    
    def __init__(self, c_eff: int, temperature: float = 1.0):
        super().__init__()
        self.score_conv = nn.Conv2d(c_eff, c_eff, kernel_size=1, bias=True)
        self.register_buffer('frozen_idx', None, persistent=True)
        self.tau = temperature

    def forward(self, x: torch.Tensor, pdim: int, training_mode: bool = True):
        B, C, H, W = x.shape
        if (not training_mode) and (self.frozen_idx is not None):
            idx = self.frozen_idx
            x_sel = torch.index_select(x, 1, idx)
            return x_sel, idx.expand(B, -1)
        
        s = self.score_conv(x)
        s = F.adaptive_avg_pool2d(s, 1).flatten(1)
        g = -torch.empty_like(s).exponential_().log()
        logits = (s + g) / max(self.tau, 1e-4)
        topk = torch.topk(logits, k=pdim, dim=-1)
        idx = topk.indices
        
        x_flat = x.view(B, C, -1)
        out = torch.gather(x_flat, 1, idx.unsqueeze(-1).expand(B, pdim, H * W))
        out = out.view(B, pdim, H, W)
        return out, idx


class TokenMerger(nn.Module):
    """Token merging module for reducing computational cost."""
    
    def __init__(self, merge_ratio: float = 0.5):
        super().__init__()
        self.merge_ratio = merge_ratio
    
    def forward(self, x: torch.Tensor, H: int, W: int):
        B, N, C = x.shape
        if self.merge_ratio >= 0.99:
            return x, H, W
        
        win = 2 if self.merge_ratio <= 0.5 else 1
        if win > 1:
            x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2)
            x_pooled = F.avg_pool2d(x_2d, kernel_size=win, stride=win)
            H_new, W_new = x_pooled.shape[-2:]
            x_merged = x_pooled.flatten(2).transpose(1, 2)
            return x_merged, H_new, W_new
        
        return x, H, W


class HiLoAttention(nn.Module):
    """HiLo attention module combining local and global attention."""
    
    def __init__(self, dim: int, alpha: float = 0.6, window_size: int = 2):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.window_size = window_size
        self.hi_dim = max(1, int(alpha * dim))
        self.lo_dim = dim - self.hi_dim
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, C, H, W = q.shape
        
        q_hi, q_lo = q[:, :self.hi_dim], q[:, self.hi_dim:]
        k_hi, k_lo = k[:, :self.hi_dim], k[:, self.hi_dim:]
        v_hi, v_lo = v[:, :self.hi_dim], v[:, self.hi_dim:]
        
        out_hi = self._local_attention(q_hi, k_hi, v_hi)
        
        out_lo = self._global_attention(q_lo, k_lo, v_lo)
        
        out = torch.cat([out_hi, out_lo], dim=1)
        return out
    
    def _local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, C, H, W = v.shape
        kernel_size = 3
        v_local = F.avg_pool2d(F.pad(v, (1, 1, 1, 1)), kernel_size=kernel_size, stride=1)
        return v_local
    
    def _global_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, C, H, W = q.shape
        k_global = F.adaptive_avg_pool2d(k, (H//2, W//2))
        v_global = F.adaptive_avg_pool2d(v, (H//2, W//2))
        
        q_flat = q.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        k_flat = k_global.view(B, C, -1).transpose(1, 2)  # [B, HW/4, C]
        v_flat = v_global.view(B, C, -1).transpose(1, 2)  # [B, HW/4, C]
        
        attn = torch.bmm(q_flat, k_flat.transpose(1, 2)) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        out_flat = torch.bmm(attn, v_flat)  # [B, HW, C]
        out = out_flat.transpose(1, 2).view(B, C, H, W)
        return out


class HRAnchorFusion(nn.Module):
    """HR-Anchor fusion module for small object detection."""
    
    def __init__(self, in_channels: int, anchor_dim: int = 32, num_anchors: int = 8):
        super().__init__()
        self.num_anchors = num_anchors
        self.anchor_dim = anchor_dim
        
        self.anchor_conv = nn.Sequential(
            nn.Conv2d(in_channels, anchor_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(anchor_dim, anchor_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.cross_attn = nn.MultiheadAttention(anchor_dim, num_heads=1, batch_first=True)
        
    def forward(self, x: torch.Tensor, stage1_features: torch.Tensor):
        B, C, H, W = x.shape
        
        anchors = self.anchor_conv(stage1_features)
        anchors_flat = anchors.flatten(2).transpose(1, 2)  # [B, N, anchor_dim]
        
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
        
        return x


class SHSABlock(nn.Module):
    """Single-Head Self-Attention block with DASH-HiLo modifications."""
    
    def __init__(self, dim: int, enable_dash: bool = False, enable_hilo: bool = False):
        super().__init__()
        self.dim = dim
        self.enable_dash = enable_dash
        self.enable_hilo = enable_hilo
        
        self.norm1 = nn.BatchNorm2d(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        
        if enable_dash:
            self.channel_scorer = ChannelScorer(dim)
            self.token_merger = TokenMerger(merge_ratio=0.3)
        
        if enable_hilo:
            self.hilo_attn = HiLoAttention(dim)
        
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 4, dim, 1)
        )
    
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        shortcut = x
        
        x = self.norm1(x)
        
        if self.enable_dash:
            pdim = max(1, int(0.7 * C))  # Use 70% of channels
            x_sel, _ = self.channel_scorer(x, pdim, training_mode=self.training)
            actual_channels = x_sel.shape[1]
            if actual_channels != self.qkv.in_channels:
                temp_qkv = nn.Conv2d(actual_channels, actual_channels * 3, 1, bias=False).to(x_sel.device)
                with torch.no_grad():
                    if actual_channels <= self.qkv.in_channels:
                        temp_qkv.weight.data = self.qkv.weight.data[:actual_channels*3, :actual_channels, :, :]
                    else:
                        temp_qkv.weight.data[:self.qkv.out_channels, :self.qkv.in_channels, :, :] = self.qkv.weight.data
                qkv = temp_qkv(x_sel)
            else:
                qkv = self.qkv(x_sel)
        else:
            x_sel = x
            qkv = self.qkv(x_sel)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        if self.enable_hilo:
            attn_out = self.hilo_attn(q, k, v)
        else:
            q_flat = q.flatten(2).transpose(1, 2)
            k_flat = k.flatten(2).transpose(1, 2)
            v_flat = v.flatten(2).transpose(1, 2)
            
            attn = torch.bmm(q_flat, k_flat.transpose(1, 2)) / math.sqrt(q_flat.shape[-1])
            attn = F.softmax(attn, dim=-1)
            out_flat = torch.bmm(attn, v_flat)
            attn_out = out_flat.transpose(1, 2).view(B, -1, H, W)
        
        if self.enable_dash and attn_out.shape[1] != C:
            if attn_out.shape[1] < C:
                pad = C - attn_out.shape[1]
                attn_out = F.pad(attn_out, (0, 0, 0, 0, 0, pad))
            else:
                attn_out = attn_out[:, :C]
        
        x = self.proj(attn_out)
        x = shortcut + x
        
        x = x + self.ffn(self.norm2(x))
        
        return x


class DASHHiLoSHViT(nn.Module):
    """DASH-HiLo-Anchor SHViT model."""
    
    def __init__(self, num_classes: int = 4, img_size: int = 96, 
                 enable_dash: bool = True, enable_hilo: bool = True, enable_anchor: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.enable_anchor = enable_anchor
        
        self.patch_embed = nn.Conv2d(3, 64, kernel_size=4, stride=4)
        
        dims = [64, 128, 256, 512]
        self.stages = nn.ModuleList()
        
        for i, dim in enumerate(dims):
            if i > 0:
                downsample = nn.Conv2d(dims[i-1], dim, kernel_size=2, stride=2)
                self.stages.append(downsample)
            
            blocks = nn.ModuleList([
                SHSABlock(dim, enable_dash=enable_dash, enable_hilo=enable_hilo)
                for _ in range(2)
            ])
            self.stages.append(blocks)
        
        if enable_anchor:
            self.hr_anchor = HRAnchorFusion(dims[0], anchor_dim=32)
        
        self.norm = nn.BatchNorm2d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        
        x = self.patch_embed(x)  # [B, 64, 24, 24]
        stage1_features = x
        
        for i, stage in enumerate(self.stages):
            if isinstance(stage, nn.Conv2d):
                x = stage(x)
            else:
                for block in stage:
                    x = block(x)
        
        if self.enable_anchor and hasattr(self, 'hr_anchor'):
            x = self.hr_anchor(x, stage1_features)
        
        x = self.norm(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.head(x)
        
        return x


def train_model(model, train_loader, val_loader, config, model_name="model"):
    """Train a model and return training history."""
    
    device = config['device']
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"Training {model_name} model...")
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 
                      os.path.join(config['save_dir'], f'{model_name}_best.pth'))
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return model, history


def plot_training_curves(histories, save_path):
    """Plot training curves for all models."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    for name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax1.plot(epochs, history['train_loss'], label=f'{name} train')
        ax2.plot(epochs, history['val_loss'], label=f'{name} val')
        ax3.plot(epochs, history['train_acc'], label=f'{name} train')
        ax4.plot(epochs, history['val_acc'], label=f'{name} val')
    
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True)
    
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_path}")


if __name__ == "__main__":
    from preprocess import create_datasets
    
    print("Testing training script...")
    
    train_loader, val_loader, _ = create_datasets(batch_size=16)
    
    model = DASHHiLoSHViT(num_classes=4)
    
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_epochs': 2,
        'learning_rate': 1e-3,
        'save_dir': 'models'
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    model, history = train_model(model, train_loader, val_loader, config, "test")
    
    print("Training test complete!")
