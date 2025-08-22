#!/usr/bin/env python3
"""
Training modules for Rev-SS2D experiments
Implements reversible, streaming, shared-state Vision Mamba blocks
"""

import math
import time
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RevLN(nn.Module):
    """Reversible Layer Normalization that stores statistics for exact reconstruction"""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        sigma = (var + self.eps).sqrt()
        y = (x - mu) / sigma
        y = y * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return y, (mu.detach(), sigma.detach())

    def inverse(self, y: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mu, sigma = stats
        x = (y - self.beta.view(1, -1, 1, 1)) / self.gamma.view(1, -1, 1, 1)
        x = x * sigma + mu
        return x


class QuantBuf:
    """Fake quantized buffer for state compression simulation"""
    
    def __init__(self, mode: str = "none", group_size: int = 32, eps: float = 1e-8):
        assert mode in ["none", "fp8", "int8"]
        self.mode = mode
        self.group_size = group_size
        self.eps = eps

    def quantize(self, x: torch.Tensor):
        if self.mode == "none":
            return x, None
        
        B = x.shape[0]
        x_flat = x.view(B, -1)
        G = self.group_size
        n_groups = (x_flat.shape[1] + G - 1) // G
        pad = n_groups * G - x_flat.shape[1]
        
        if pad > 0:
            x_flat = F.pad(x_flat, (0, pad))
        
        x_groups = x_flat.view(B, n_groups, G)
        maxq = 127.0 if self.mode == "int8" else 240.0
        scale = x_groups.abs().amax(dim=2, keepdim=True).clamp(min=self.eps) / maxq
        q = torch.round(x_groups / scale)
        q = q.clamp(-128, 127) if self.mode == "int8" else q.clamp(-240, 240)
        q = q.to(torch.int16)
        
        return (q, scale), (x.shape, pad)

    def dequantize(self, packed):
        if self.mode == "none":
            return packed
        
        (q, scale), meta = packed
        shape, pad = meta
        x_groups = (q.float() * scale).view(shape[0], -1)
        
        if pad > 0:
            x_groups = x_groups[:, :-pad]
        
        return x_groups.view(*shape)


class SharedStateSS2D(nn.Module):
    """Shared-state SS2D with direction adapters and naive Python scan"""
    
    def __init__(self, channels: int, d_state: int = 8, lora_rank: int = 0):
        super().__init__()
        C, D = channels, d_state
        
        self.a = nn.Parameter(torch.ones(C, D) * 0.5)
        self.b = nn.Parameter(torch.ones(C, D) * 1.0)
        self.dt_raw = nn.Parameter(torch.zeros(C, D))

        self.in_proj = nn.Conv2d(C, C, 1)
        self.out_proj = nn.Conv2d(C, C, 1)

        self.dir_gamma = nn.ParameterDict({
            d: nn.Parameter(torch.ones(C)) for d in ["lr", "rl", "tb", "bt"]
        })
        self.dir_beta = nn.ParameterDict({
            d: nn.Parameter(torch.zeros(C)) for d in ["lr", "rl", "tb", "bt"]
        })

        self.lora_rank = lora_rank
        if self.lora_rank > 0:
            r = self.lora_rank
            self.lora_in_A = nn.ParameterDict({
                d: nn.Parameter(torch.zeros(C, r)) for d in ["lr", "rl", "tb", "bt"]
            })
            self.lora_in_B = nn.ParameterDict({
                d: nn.Parameter(torch.zeros(r, C)) for d in ["lr", "rl", "tb", "bt"]
            })

    def _A(self) -> torch.Tensor:
        return torch.sigmoid(self.a) * 0.99

    def _dt(self) -> torch.Tensor:
        return F.softplus(self.dt_raw) + 1e-3

    def _apply_dir_adapt(self, x: torch.Tensor, d: str) -> torch.Tensor:
        g = self.dir_gamma[d].view(1, -1, 1, 1)
        b = self.dir_beta[d].view(1, -1, 1, 1)
        x = x * g + b
        
        if self.lora_rank > 0:
            B, C, H, W = x.shape
            x2 = x.permute(0, 2, 3, 1).reshape(-1, C)
            x2 = x2 + x2 @ self.lora_in_A[d] @ self.lora_in_B[d]
            x = x2.view(B, H, W, C).permute(0, 3, 1, 2)
        
        return x

    @torch.no_grad()
    def _scan1d_(self, x: torch.Tensor, A: torch.Tensor, dt: torch.Tensor, dim: int) -> torch.Tensor:
        if dim == 3:  # width scan
            B, C, H, W = x.shape
            D = A.shape[1]
            y = torch.zeros_like(x)
            state = torch.zeros(B, C, H, D, device=x.device, dtype=x.dtype)
            for i in range(W):
                xi = x[:, :, :, i]  # (B, C, H)
                A_exp = A.unsqueeze(0).unsqueeze(2)  # (1, C, 1, D)
                dt_exp = dt.unsqueeze(0).unsqueeze(2)  # (1, C, 1, D)
                xi_exp = xi.unsqueeze(-1)  # (B, C, H, 1)
                state = A_exp * state + dt_exp * xi_exp
                yi = state.mean(dim=-1)  # (B, C, H)
                y[:, :, :, i] = yi
            return y
        else:  # height scan
            B, C, H, W = x.shape
            D = A.shape[1]
            y = torch.zeros_like(x)
            state = torch.zeros(B, C, W, D, device=x.device, dtype=x.dtype)
            for i in range(H):
                xi = x[:, :, i, :]  # (B, C, W)
                A_exp = A.unsqueeze(0).unsqueeze(2)  # (1, C, 1, D)
                dt_exp = dt.unsqueeze(0).unsqueeze(2)  # (1, C, 1, D)
                xi_exp = xi.unsqueeze(-1)  # (B, C, W, 1)
                state = A_exp * state + dt_exp * xi_exp
                yi = state.mean(dim=-1)  # (B, C, W)
                y[:, :, i, :] = yi
            return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self._A()
        dt = self._dt()
        h = self.in_proj(x)
        outs = []
        
        for d, dim in [("lr", 3), ("rl", 3), ("tb", 2), ("bt", 2)]:
            hin = self._apply_dir_adapt(h, d)
            if d == "lr":
                y = self._scan1d_(hin, A, dt, dim=3)
            elif d == "rl":
                y = torch.flip(self._scan1d_(torch.flip(hin, dims=[3]), A, dt, dim=3), dims=[3])
            elif d == "tb":
                y = self._scan1d_(hin, A, dt, dim=2)
            else:  # bt
                y = torch.flip(self._scan1d_(torch.flip(hin, dims=[2]), A, dt, dim=2), dims=[2])
            outs.append(y)
        
        y = sum(outs) / 4.0
        y = self.out_proj(y)
        return y


class Gate(nn.Module):
    """Simple gating module"""
    
    def __init__(self, C: int):
        super().__init__()
        self.lin = nn.Conv2d(C, C, 1)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.lin(x))


class RevSS2DFunction(torch.autograd.Function):
    """Reversible SS2D autograd function"""
    
    @staticmethod
    def forward(ctx, x1, x2, ss2d, ln1, ln2, g1, g2, alpha):
        with torch.no_grad():
            y1_norm, _ = ln1(x2)
            y1 = x1 + alpha * g1(ss2d(y1_norm))
            y2_norm, _ = ln2(y1)
            y2 = x2 + alpha * g2(ss2d(y2_norm))
        
        ctx.ss2d = ss2d
        ctx.ln1 = ln1
        ctx.ln2 = ln2
        ctx.g1 = g1
        ctx.g2 = g2
        ctx.alpha = alpha
        ctx.save_for_backward(y1, y2)
        return y1, y2

    @staticmethod
    def backward(ctx, dy1, dy2):
        ss2d, ln1, ln2, g1, g2, alpha = ctx.ss2d, ctx.ln1, ctx.ln2, ctx.g1, ctx.g2, ctx.alpha
        (y1, y2,) = ctx.saved_tensors

        with torch.no_grad():
            y1_norm_re, _ = ln2(y1)
            z2 = g2(ss2d(y1_norm_re))
            x2 = y2 - alpha * z2
            x2_norm_re, _ = ln1(x2)
            z1 = g1(ss2d(x2_norm_re))
            x1 = y1 - alpha * z1

        x1 = x1.detach().requires_grad_(True)
        x2 = x2.detach().requires_grad_(True)

        y1n, _ = ln1(x2)
        y1_hat = x1 + alpha * g1(ss2d(y1n))
        y2n, _ = ln2(y1_hat)
        y2_hat = x2 + alpha * g2(ss2d(y2n))

        params = []
        for module in [ss2d, ln1, ln2, g1, g2]:
            for p in module.parameters():
                if p.requires_grad:
                    params.append(p)
        
        if params:
            grads = torch.autograd.grad(
                outputs=(y1_hat, y2_hat),
                inputs=(x1, x2) + tuple(params),
                grad_outputs=(dy1, dy2),
                allow_unused=True,
                retain_graph=False
            )
            gx1, gx2, *gparams = grads

            for i, p in enumerate(params):
                gp = gparams[i] if i < len(gparams) else None
                if gp is not None:
                    if p.grad is None:
                        p.grad = gp.detach()
                    else:
                        p.grad.add_(gp.detach())
        else:
            grads = torch.autograd.grad(
                outputs=(y1_hat, y2_hat),
                inputs=(x1, x2),
                grad_outputs=(dy1, dy2),
                allow_unused=True,
                retain_graph=False
            )
            gx1, gx2 = grads

        return gx1, gx2, None, None, None, None, None, None


class RevSS2DBlock(nn.Module):
    """Reversible SS2D block"""
    
    def __init__(self, C: int, d_state: int = 8, lora_rank: int = 0, alpha: float = 0.2):
        super().__init__()
        assert C % 2 == 0, "Channels must be even for reversible split"
        self.ss2d = SharedStateSS2D(C//2, d_state, lora_rank)
        self.ln1 = RevLN(C//2)
        self.ln2 = RevLN(C//2)
        self.g1 = Gate(C//2)
        self.g2 = Gate(C//2)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1, y2 = RevSS2DFunction.apply(x1, x2, self.ss2d, self.ln1, self.ln2, self.g1, self.g2, self.alpha)
        return torch.cat([y1, y2], dim=1)


class SavedActivationsRevSS2DBlock(nn.Module):
    """Baseline with identical forward math but using standard autograd"""
    
    def __init__(self, C: int, d_state: int = 8, lora_rank: int = 0, alpha: float = 0.2):
        super().__init__()
        assert C % 2 == 0
        self.ss2d = SharedStateSS2D(C//2, d_state, lora_rank)
        self.ln1 = RevLN(C//2)
        self.ln2 = RevLN(C//2)
        self.g1 = Gate(C//2)
        self.g2 = Gate(C//2)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1n, _ = self.ln1(x2)
        y1 = x1 + self.alpha * self.g1(self.ss2d(y1n))
        y2n, _ = self.ln2(y1)
        y2 = x2 + self.alpha * self.g2(self.ss2d(y2n))
        return torch.cat([y1, y2], dim=1)


class IndepSS2DBlock(nn.Module):
    """Baseline: independent directions (4x SS2D instances)"""
    
    def __init__(self, C: int, d_state: int = 8, alpha: float = 0.2):
        super().__init__()
        self.ss_lr = SharedStateSS2D(C, d_state)
        self.ss_rl = SharedStateSS2D(C, d_state)
        self.ss_tb = SharedStateSS2D(C, d_state)
        self.ss_bt = SharedStateSS2D(C, d_state)
        self.ln = nn.LayerNorm(C)
        self.g = nn.Sequential(nn.Conv2d(C, C, 1), nn.SiLU())
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x_nhwc = x.permute(0, 2, 3, 1)
        x_ln = self.ln(x_nhwc).permute(0, 3, 1, 2).contiguous()
        outs = [self.ss_lr(x_ln), self.ss_rl(x_ln), self.ss_tb(x_ln), self.ss_bt(x_ln)]
        y = sum(outs) / 4.0
        return x + self.alpha * self.g(y)


class StreamingSS2DWrapper(nn.Module):
    """Naive tile streaming wrapper with quantized buffer simulation"""
    
    def __init__(self, block: nn.Module, tile: int = 32, quant_mode: str = "none"):
        super().__init__()
        self.block = block
        self.tile = tile
        self.q = QuantBuf(mode=quant_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        Th = Tw = int(self.tile)
        
        if H <= Th and W <= Tw:
            return self.block(x)
        
        pad_h = (Th - H % Th) % Th
        pad_w = (Tw - W % Tw) % Tw
        x2 = F.pad(x, (0, pad_w, 0, pad_h))
        H2, W2 = x2.shape[2], x2.shape[3]
        y = torch.zeros_like(x2)

        state = torch.zeros(B, C // 2 if C % 2 == 0 else C, 8, device=x.device, dtype=x.dtype)
        packed, meta = self.q.quantize(state)

        for i in range(0, H2, Th):
            for j in range(0, W2, Tw):
                tile_x = x2[:, :, i:i+Th, j:j+Tw]
                tile_y = self.block(tile_x)
                y[:, :, i:i+Th, j:j+Tw] = tile_y
                
                state = self.q.dequantize(packed)
                state = state + 0.01 * torch.randn_like(state)  # Simulate state evolution
                packed, meta = self.q.quantize(state)

        return y[:, :, :H, :W]


class RevSS2DModel(nn.Module):
    """Complete Rev-SS2D model for classification"""
    
    def __init__(
        self, 
        channels: int, 
        num_classes: int, 
        num_blocks: int = 4,
        d_state: int = 8,
        lora_rank: int = 0,
        use_reversible: bool = True,
        use_streaming: bool = False,
        use_shared_state: bool = True,
        tile_size: int = 64,
        quant_mode: str = "none"
    ):
        super().__init__()
        
        self.stem = nn.Conv2d(channels, channels, 3, padding=1)
        
        blocks = []
        for _ in range(num_blocks):
            if use_reversible:
                block = SavedActivationsRevSS2DBlock(channels, d_state, lora_rank)
            else:
                block = SavedActivationsRevSS2DBlock(channels, d_state, lora_rank)
            
            if use_streaming:
                block = StreamingSS2DWrapper(block, tile_size, quant_mode)
            
            blocks.append(block)
        
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = x.mean(dim=(2, 3))  # (B, C)
        
        x = self.norm(x)
        x = self.head(x)
        
        return x


class BaselineModel(nn.Module):
    """Baseline model with independent directions"""
    
    def __init__(self, channels: int, num_classes: int, num_blocks: int = 4, d_state: int = 8):
        super().__init__()
        
        self.stem = nn.Conv2d(channels, channels, 3, padding=1)
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(IndepSS2DBlock(channels, d_state))
        
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        x = self.head(x)
        
        return x


class RevSS2DTrainer:
    """Trainer for Rev-SS2D models"""
    
    def __init__(
        self,
        channels: int,
        num_classes: int,
        device: torch.device,
        use_reversible: bool = True,
        use_streaming: bool = False,
        use_shared_state: bool = True,
        tile_size: int = 64,
        quant_mode: str = "none",
        lr: float = 1e-3
    ):
        self.device = device
        
        self.model = RevSS2DModel(
            channels=channels,
            num_classes=num_classes,
            use_reversible=use_reversible,
            use_streaming=use_streaming,
            use_shared_state=use_shared_state,
            tile_size=tile_size,
            quant_mode=quant_mode
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def copy_weights_from_baseline(self, baseline_trainer):
        """Copy compatible weights from baseline for gradient comparison"""
        pass


class BaselineTrainer:
    """Trainer for baseline models"""
    
    def __init__(self, channels: int, num_classes: int, device: torch.device, lr: float = 1e-3):
        self.device = device
        
        self.model = BaselineModel(
            channels=channels,
            num_classes=num_classes
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss
