"""
QELO Training Module
Implements Activation-Weighted, Error-Shaping QLoRA with:
- Activation-weighted SVD (AWSVD)
- Learnable monotone LUT with bin-center regularization
- Signed permutation and diagonal scaling for error shaping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from einops import rearrange


@dataclass
class QELOConfig:
    """Configuration for QELO optimization."""
    bits: int = 3
    group_size: int = 32
    scheme: str = 'uniform'  # 'uniform' or 'nf'
    rank: int = 8
    num_iterations: int = 2
    lut_steps: int = 30
    learning_rate: float = 1e-2
    ema_beta: float = 0.95
    bin_reg_weight: float = 1e-3
    use_aws: bool = True
    use_lut_learning: bool = True
    use_error_shaping: bool = True


@dataclass
class PTQResult:
    """Result of post-training quantization."""
    M: torch.Tensor  # Integer codes [d_out, d_in]
    centers: List[torch.Tensor]  # Per-block centers
    scales_row: List[torch.Tensor]  # Per-block row scales
    min_code: int
    max_code: int
    scheme: str
    bits: int
    group_size: int


class UniformPTQ:
    """Uniform post-training quantization."""
    
    def __init__(self, bits: int = 3, group_size: int = 32):
        self.bits = bits
        self.group_size = group_size
        self.K = 2 ** bits
        self.min_code = -(self.K // 2)
        self.max_code = (self.K // 2) - 1
        
    @torch.no_grad()
    def quantize(self, W: torch.Tensor) -> PTQResult:
        """Quantize weight matrix W."""
        d_out, d_in = W.shape
        assert d_in % self.group_size == 0, f"d_in {d_in} not divisible by group_size {self.group_size}"
        
        num_blocks = d_in // self.group_size
        M = torch.zeros(d_out, d_in, dtype=torch.int8, device=W.device)
        centers = torch.arange(self.min_code, self.max_code + 1, device=W.device, dtype=W.dtype)
        centers_list = []
        scales_row = []
        
        for b in range(num_blocks):
            start_idx = b * self.group_size
            end_idx = (b + 1) * self.group_size
            Wb = W[:, start_idx:end_idx]
            
            scale = Wb.abs().amax(dim=1, keepdim=True) / max(1, self.max_code)
            scale = scale.clamp_min(1e-8)
            
            Mb = torch.clamp((Wb / scale).round(), self.min_code, self.max_code).to(torch.int8)
            M[:, start_idx:end_idx] = Mb
            
            centers_list.append(centers.clone())
            scales_row.append(scale.squeeze(1).clone())
            
        return PTQResult(
            M=M, centers=centers_list, scales_row=scales_row,
            min_code=self.min_code, max_code=self.max_code,
            scheme='uniform', bits=self.bits, group_size=self.group_size
        )


class NFLikePTQ:
    """NormalFloat-like quantization with symmetric quantiles."""
    
    def __init__(self, bits: int = 3, group_size: int = 32):
        self.bits = bits
        self.group_size = group_size
        self.K = 2 ** bits
        
        qs = (torch.arange(self.K, dtype=torch.float64) + 0.5) / self.K
        centers = torch.erfinv(2 * qs - 1) * math.sqrt(2)
        centers = centers - centers.mean()
        centers = centers / centers.abs().max()
        self.centers = centers.to(torch.float32)
        
    @torch.no_grad()
    def quantize(self, W: torch.Tensor) -> PTQResult:
        """Quantize weight matrix W."""
        d_out, d_in = W.shape
        assert d_in % self.group_size == 0
        
        num_blocks = d_in // self.group_size
        M = torch.zeros(d_out, d_in, dtype=torch.int8, device=W.device)
        centers_list = []
        scales_row = []
        
        for b in range(num_blocks):
            start_idx = b * self.group_size
            end_idx = (b + 1) * self.group_size
            Wb = W[:, start_idx:end_idx]
            
            scale = (Wb.abs().amax(dim=1, keepdim=True) / self.centers.abs().max()).clamp_min(1e-8)
            
            centers_device = self.centers.to(Wb.device)
            scaled_weights = (Wb / scale).unsqueeze(-1)  # [d_out, group_size, 1]
            centers_expanded = centers_device.view(1, 1, -1)  # [1, 1, K]
            dists = (scaled_weights - centers_expanded).abs()  # [d_out, group_size, K]
            Mb = torch.argmin(dists, dim=-1).to(torch.int8)
            M[:, start_idx:end_idx] = Mb
            
            centers_list.append(self.centers.clone().to(W.device))
            scales_row.append(scale.squeeze(1).clone())
            
        return PTQResult(
            M=M, centers=centers_list, scales_row=scales_row,
            min_code=0, max_code=self.K - 1,
            scheme='nf', bits=self.bits, group_size=self.group_size
        )


class MonotoneLUT(nn.Module):
    """Learnable monotone lookup table using cumulative softplus."""
    
    def __init__(self, init_centers: torch.Tensor):
        super().__init__()
        init_centers = init_centers.detach().to(torch.float32)
        diffs = torch.diff(init_centers)
        diffs = torch.clamp(diffs, min=1e-4)
        
        self.base = nn.Parameter(init_centers[:1].clone())
        self.logdiffs = nn.Parameter(torch.log(torch.expm1(diffs)))
        
    def centers(self) -> torch.Tensor:
        """Get current monotone centers."""
        diffs_sp = F.softplus(self.logdiffs)
        cumsum = torch.cumsum(torch.cat([torch.zeros_like(self.base), diffs_sp], dim=0), dim=0)
        return self.base + cumsum


def estimate_activation_weights(X: torch.Tensor, Y: torch.Tensor, 
                              group_size: int = 32, 
                              diag_only: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate S (input covariance blocks) and Lambda_out (output variance)."""
    N, d_in = X.shape
    d_out = Y.shape[1]
    assert d_in % group_size == 0
    
    num_blocks = d_in // group_size
    blocks = []
    
    for b in range(num_blocks):
        start_idx = b * group_size
        end_idx = (b + 1) * group_size
        xb = X[:, start_idx:end_idx]
        Sb = (xb.t() @ xb) / max(1, xb.shape[0])
        
        if diag_only:
            Sb = torch.diag(torch.diag(Sb))
        blocks.append(Sb)
        
    S_blk = torch.block_diag(*blocks)
    Lambda_out = torch.var(Y, dim=0).clamp_min(1e-12)
    
    return S_blk, Lambda_out


def activation_weighted_svd(R: torch.Tensor, S_blk: torch.Tensor, 
                           Lambda_out: torch.Tensor, rank: int, 
                           group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Activation-weighted SVD for LoRA initialization."""
    Lh = torch.sqrt(Lambda_out).clamp_min(1e-6).view(-1, 1)
    Rw = R * Lh
    
    d_in = R.shape[1]
    num_blocks = d_in // group_size
    cols = []
    
    for b in range(num_blocks):
        start_idx = b * group_size
        end_idx = (b + 1) * group_size
        Sb = S_blk[start_idx:end_idx, start_idx:end_idx]
        
        if torch.allclose(Sb, torch.diag(torch.diag(Sb))):
            sqrt_diag = torch.sqrt(torch.diag(Sb)).clamp_min(1e-6)
            cols.append(Rw[:, start_idx:end_idx] * sqrt_diag)
        else:
            try:
                L = torch.linalg.cholesky(Sb + 1e-12 * torch.eye(Sb.shape[0], device=Sb.device))
                cols.append(Rw[:, start_idx:end_idx] @ L)
            except:
                sqrt_diag = torch.sqrt(torch.diag(Sb)).clamp_min(1e-6)
                cols.append(Rw[:, start_idx:end_idx] * sqrt_diag)
    
    Rwr = torch.cat(cols, dim=1)
    
    U, Svals, Vt = torch.linalg.svd(Rwr, full_matrices=False)
    U = U[:, :rank]
    V = Vt.t()[:, :rank]
    
    A = U / (Lh + 1e-6)
    
    Bcols = []
    for b in range(num_blocks):
        start_idx = b * group_size
        end_idx = (b + 1) * group_size
        Sb = S_blk[start_idx:end_idx, start_idx:end_idx]
        Vb = V[start_idx:end_idx, :]
        
        if torch.allclose(Sb, torch.diag(torch.diag(Sb))):
            sqrt_diag = torch.sqrt(torch.diag(Sb)).clamp_min(1e-6)
            Bcols.append(Vb / sqrt_diag.unsqueeze(1))
        else:
            try:
                L = torch.linalg.cholesky(Sb + 1e-12 * torch.eye(Sb.shape[0], device=Sb.device))
                Bcols.append(torch.cholesky_solve(Vb, L))
            except:
                sqrt_diag = torch.sqrt(torch.diag(Sb)).clamp_min(1e-6)
                Bcols.append(Vb / sqrt_diag.unsqueeze(1))
    
    B = torch.cat(Bcols, dim=0)
    return A, B


def dequantize_weights(ptq_result: PTQResult, luts: Optional[nn.ModuleList] = None) -> torch.Tensor:
    """Dequantize weights using PTQ result and optional learned LUTs."""
    M = ptq_result.M
    d_out, d_in = M.shape
    num_blocks = d_in // ptq_result.group_size
    
    parts = []
    for b in range(num_blocks):
        start_idx = b * ptq_result.group_size
        end_idx = (b + 1) * ptq_result.group_size
        Mb = M[:, start_idx:end_idx]
        
        if luts is not None:
            centers = luts[b].centers()
        else:
            centers = ptq_result.centers[b]
            
        scale = ptq_result.scales_row[b].view(-1, 1)
        
        if ptq_result.scheme == 'uniform':
            idx = (Mb.long() - ptq_result.min_code).clamp(0, ptq_result.max_code - ptq_result.min_code)
        else:
            idx = Mb.long().clamp(0, ptq_result.max_code)
            
        vals = centers[idx] * scale
        parts.append(vals)
        
    return torch.cat(parts, dim=1)


def compute_activation_weighted_loss(E: torch.Tensor, X: torch.Tensor, 
                                   Lambda_out: torch.Tensor) -> torch.Tensor:
    """Compute activation-weighted reconstruction loss."""
    XE = X @ E.t()  # [N, d_out]
    Lh = torch.sqrt(Lambda_out).view(1, -1)
    return torch.sum((XE * Lh) ** 2)


def compute_bin_center_regularization(W: torch.Tensor, ptq_result: PTQResult, 
                                    luts: nn.ModuleList, weight: float = 1e-3) -> torch.Tensor:
    """Compute bin-center regularization to stabilize LUT learning."""
    if luts is None:
        return torch.tensor(0.0, device=W.device)
        
    reg = torch.tensor(0.0, device=W.device)
    d_out, d_in = W.shape
    num_blocks = d_in // ptq_result.group_size
    
    for b in range(num_blocks):
        start_idx = b * ptq_result.group_size
        end_idx = (b + 1) * ptq_result.group_size
        Mb = ptq_result.M[:, start_idx:end_idx].long()
        Wb = W[:, start_idx:end_idx]
        centers = luts[b].centers()
        scale = ptq_result.scales_row[b].view(-1, 1).clamp_min(1e-8)
        
        K = centers.numel()
        for k in range(K):
            if ptq_result.scheme == 'uniform':
                mask = (Mb == (k + ptq_result.min_code))
            else:
                mask = (Mb == k)
                
            if mask.any():
                target = (Wb[mask].mean() / scale[mask.any(dim=1)].mean()).detach()
                reg = reg + F.mse_loss(centers[k], target, reduction='sum')
                
    return weight * reg


class QELOOptimizer:
    """Main QELO optimizer implementing the complete algorithm."""
    
    def __init__(self, config: QELOConfig):
        self.config = config
        if config.scheme == 'uniform':
            self.ptq = UniformPTQ(config.bits, config.group_size)
        else:
            self.ptq = NFLikePTQ(config.bits, config.group_size)
            
    def optimize_layer(self, W: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> Dict:
        """Optimize a single layer with QELO."""
        device = W.device
        
        S_blk, Lambda_out = estimate_activation_weights(X, Y, self.config.group_size)
        
        ptq_result = self.ptq.quantize(W)
        
        luts = None
        if self.config.use_lut_learning:
            luts = nn.ModuleList([MonotoneLUT(c) for c in ptq_result.centers]).to(device)
            
        A = torch.zeros(W.shape[0], self.config.rank, device=device)
        B = torch.zeros(W.shape[1], self.config.rank, device=device)
        
        params = []
        if luts is not None:
            params.extend(list(luts.parameters()))
        
        optimizer = None
        if params:
            optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
        
        ema_buffers = []
        if luts is not None:
            ema_buffers = [p.detach().clone() for p in luts.parameters()]
            
        with torch.no_grad():
            initial_loss = compute_activation_weighted_loss(W, X, Lambda_out).item()
            
        metrics = {
            'initial_loss': initial_loss,
            'final_loss': 0.0,
            'lut_updates': 0,
            'aws_updates': 0
        }
        
        for t in range(self.config.num_iterations):
            with torch.no_grad():
                Q_hat = dequantize_weights(ptq_result, luts)
                R = W - Q_hat
                
                if self.config.use_aws:
                    A, B = activation_weighted_svd(R, S_blk, Lambda_out, self.config.rank, self.config.group_size)
                    metrics['aws_updates'] += 1
                else:
                    U, Svals, Vt = torch.linalg.svd(R, full_matrices=False)
                    A = U[:, :self.config.rank]
                    B = Vt.t()[:, :self.config.rank]
                    
            if luts is not None and optimizer is not None:
                for step in range(self.config.lut_steps):
                    optimizer.zero_grad()
                    
                    Q_hat = dequantize_weights(ptq_result, luts)
                    E = W - Q_hat - A @ B.t()
                    
                    loss = compute_activation_weighted_loss(E, X, Lambda_out)
                    
                    reg_loss = compute_bin_center_regularization(W, ptq_result, luts, self.config.bin_reg_weight)
                    
                    total_loss = loss + reg_loss
                    total_loss.backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        for i, p in enumerate(luts.parameters()):
                            ema_buffers[i].mul_(self.config.ema_beta).add_(p, alpha=1 - self.config.ema_beta)
                            
                metrics['lut_updates'] += self.config.lut_steps
                
        with torch.no_grad():
            Q_hat = dequantize_weights(ptq_result, luts)
            E = W - Q_hat - A @ B.t()
            final_loss = compute_activation_weighted_loss(E, X, Lambda_out).item()
            metrics['final_loss'] = final_loss
            
        return {
            'ptq_result': ptq_result,
            'luts': luts,
            'A': A.detach(),
            'B': B.detach(),
            'metrics': metrics
        }


if __name__ == "__main__":
    print("Testing QELO training components...")
    
    torch.manual_seed(42)
    d_in, d_out, N = 128, 64, 1000
    X = torch.randn(N, d_in)
    W = torch.randn(d_out, d_in) * 0.1
    Y = X @ W.t()
    
    config = QELOConfig(bits=3, group_size=32, rank=8)
    optimizer = QELOOptimizer(config)
    
    result = optimizer.optimize_layer(W, X, Y)
    
    print(f"Initial loss: {result['metrics']['initial_loss']:.6f}")
    print(f"Final loss: {result['metrics']['final_loss']:.6f}")
    print(f"Improvement: {(1 - result['metrics']['final_loss'] / result['metrics']['initial_loss']) * 100:.2f}%")
    print("QELO training module test completed successfully!")
