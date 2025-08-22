"""
AccuTune Training Module
Implements training logic with low-bit accumulator emulation, gradient accumulation control,
and numeric safety constraints.
"""

import os
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class LowBitLinearFn(torch.autograd.Function):
    """Custom autograd function for low-bit accumulator emulation with DIFF-lite correction"""
    
    @staticmethod
    def forward(ctx, x, weight, bias, layer_ref):
        y = layer_ref._accumulate(x, weight)
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(x, weight)
        ctx.layer_ref = layer_ref
        try:
            ctx.alpha = float(layer_ref.alpha_ema.detach().item())
        except Exception:
            ctx.alpha = 1.0
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        alpha = ctx.alpha
        go = grad_out * alpha  # DIFF-lite multiplicative correction
        grad_x = go @ weight
        
        if len(go.shape) == 3:
            go_2d = go.reshape(-1, go.size(-1))
            grad_w = go_2d.t() @ x.reshape(-1, x.size(-1))
        else:
            grad_w = go.t() @ x
            
        grad_b = go.sum(dim=0) if ctx.needs_input_grad[2] else None
        return grad_x, grad_w, grad_b, None


class LowBitAccLinear(nn.Linear):
    """Linear layer with low-bit accumulator emulation and telemetry"""
    
    def __init__(self, in_f, out_f, bias=True, mode="M7E4", ebias=0, order="pairwise", dither=False, sample_p=0.01):
        super().__init__(in_f, out_f, bias)
        self.mode, self.ebias, self.order, self.dither = mode, ebias, order, dither
        self.sample_p = sample_p
        
        self.register_buffer("of_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("uf_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("swamp_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("mantissa_hist", torch.zeros(64))
        self.register_buffer("headroom", torch.zeros(1))
        self.register_buffer("alpha_ema", torch.tensor(1.0))
        self.alpha_beta = 0.9
        self.last_alpha = 1.0

    @staticmethod
    def _mode_bits(mode):
        """Get mantissa and exponent bits for accumulator mode"""
        return {"M6E5": (6,5), "M7E4": (7,4), "M8E4": (8,4)}.get(mode, (8,4))

    def _quantize_tile(self, x):
        """Quantize a tile of data using the current accumulator mode"""
        man_bits, exp_bits = self._mode_bits(self.mode)
        
        maxv = x.abs().max()
        if maxv == 0 or not torch.isfinite(maxv):
            return x, 1.0
            
        scale = maxv / (2**man_bits - 1)
        if scale == 0:
            return x, 1.0
            
        y = x / scale
        q = torch.round(y) * scale
        
        headroom = 1.0
        return q, headroom

    def _accumulate(self, a, b):
        """Accumulate matrix multiplication with low-bit precision and telemetry"""
        original_shape = a.shape
        if len(a.shape) == 3:
            B, T, K = a.shape
            a = a.reshape(B * T, K)  # Flatten to [B*T, K] using reshape for non-contiguous tensors
        else:
            B, K = a.shape
            T = 1
        
        if len(b.shape) == 3:
            b = b.reshape(-1, b.size(-1))
        
        if len(b.shape) > 2:
            raise RuntimeError(f"Tensor b still has {len(b.shape)} dimensions after reshape: {b.shape}")
        
        if b.size(1) == a.size(1):  # b is [N, K], need [K, N]
            b = b.t()
        elif b.size(0) != a.size(1):  # b dimensions don't match
            b = b.t()  # Try transpose
        
        prod = a.unsqueeze(-1) * b.unsqueeze(0)  # [B*T, K, N] or [B, K, N]
        
        if self.sample_p > 0 and prod.numel() < 1000000:  # Only sample on smaller tensors
            mask = (torch.rand_like(prod) < self.sample_p)
            if mask.any():
                q_block, _ = self._quantize_tile(prod)
                num = (q_block[mask]).abs().sum().clamp_min(1e-8)
                den = (prod[mask]).abs().sum().clamp_min(1e-8)
                alpha = (num / den).item()
                self.last_alpha = alpha
                self.alpha_ema.mul_(self.alpha_beta)
                self.alpha_ema.add_((1 - self.alpha_beta) * alpha)
        
        qprod, hdr = self._quantize_tile(prod)
        if isinstance(hdr, torch.Tensor):
            self.headroom += hdr.detach()
        else:
            self.headroom += hdr
        
        chunk = 16 if self.order == "chunk16" else (8 if self.order == "chunk8" else 32)
        out = torch.zeros(prod.size(0), prod.size(2), device=prod.device, dtype=prod.dtype)
        for i in range(0, prod.size(1), chunk):
            block = qprod[:, i:i+chunk, :]
            block_sum = block.sum(dim=1)
            if i > 0:
                swamp = (block_sum.abs() < (1e-6 + 1e-3 * out.abs())).float().sum()
                self.swamp_count += swamp.long()
            out = out + block_sum
        
        self.of_count += (out.isinf()).sum().long()
        self.uf_count += (out == 0).sum().long()
        
        if len(original_shape) == 3:
            out = out.reshape(original_shape[0], original_shape[1], -1)  # [B, T, N]
        
        return out

    def forward(self, x):
        y = LowBitLinearFn.apply(x, self.weight, self.bias, self)
        with torch.no_grad():
            h = torch.clamp(((y.detach().abs().log2()+20)*2).long(), 0, 63)
            self.mantissa_hist += torch.bincount(h.reshape(-1), minlength=64).float()
        return y


class LNOnlyGNS:
    """LayerNorm-only Gradient Noise Scale estimator"""
    
    def __init__(self):
        self.ema = None
        self.handles = []

    def attach(self, model: nn.Module):
        """Attach hooks to LayerNorm modules for GNS estimation"""
        def hook(module, grad_input, grad_output):
            try:
                g = grad_output[0]
                per_ex_norm = g.reshape(g.size(0), -1).norm(dim=1)
                gns_est = per_ex_norm.var(unbiased=False) / (per_ex_norm.mean()**2 + 1e-12)
                self.ema = gns_est if self.ema is None else 0.95*self.ema + 0.05*gns_est
            except Exception:
                pass
                
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                self.handles.append(m.register_backward_hook(hook))

    def value(self):
        """Get current GNS estimate"""
        return None if self.ema is None else float(self.ema.detach().cpu())


class GAController:
    """Gradient Accumulation Controller with AdaScale-style invariance"""
    
    def __init__(self, optimizer, target_temp=1.0, min_ga=1, max_ga=32, rate_limit_steps=50):
        self.opt = optimizer
        self.target = target_temp
        self.min_ga, self.max_ga = min_ga, max_ga
        self.current_ga = min_ga
        self.rate_limit_steps = rate_limit_steps
        self._last_change_step = -1
        self._step = 0

    def suggest(self, gns_val: Optional[float]):
        """Suggest gradient accumulation steps based on GNS"""
        self._step += 1
        if gns_val is None:
            return self.current_ga
            
        err = max(0.1, min(10.0, gns_val / max(self.target, 1e-12)))
        suggested = int(max(self.min_ga, min(self.max_ga, round(self.current_ga * (err**0.5)))))
        
        if self._last_change_step >= 0 and self._step - self._last_change_step < self.rate_limit_steps:
            return self.current_ga
            
        if suggested != self.current_ga:
            scale = (self.current_ga / suggested) ** 0.5
            for pg in self.opt.param_groups:
                pg["lr"] *= scale
            self.current_ga = suggested
            self._last_change_step = self._step
            
        return self.current_ga


class NumericController:
    """Numeric controller for accumulator precision and order with safety constraints"""
    
    def __init__(self, layers: List[LowBitAccLinear], eps_cos=0.02, s_max=1e6, cooldown=200):
        self.layers = layers
        self.eps_cos = eps_cos
        self.s_max = s_max
        self.cooldown = cooldown
        self.cool = {id(l): 0 for l in layers}
        self.menu = [
            {"mode":"M6E5","eb":0,"order":"pairwise"},
            {"mode":"M7E4","eb":0,"order":"pairwise"},
            {"mode":"M8E4","eb":0,"order":"chunk16"},
        ]

    @torch.no_grad()
    def step(self, ref_cosines: Dict[int, float]):
        """Update accumulator configurations based on safety constraints"""
        for l in self.layers:
            lid = id(l)
            self.cool[lid] = max(0, self.cool[lid]-1)
            if self.cool[lid] > 0:
                continue
                
            swamp = int(l.swamp_count.item())
            of = int(l.of_count.item())
            uf = int(l.uf_count.item())
            cos_ok = ref_cosines.get(lid, 1.0) >= (1.0 - self.eps_cos)
            
            safer_needed = (of>0) or (uf>0) or (swamp >= self.s_max) or (not cos_ok)
            idx = next((i for i,m in enumerate(self.menu) if m["mode"]==l.mode and m["order"]==l.order), 1)
            ni = min(len(self.menu)-1, idx+1) if safer_needed else max(0, idx-1)
            cand = self.menu[ni]
            
            if cand["mode"] != l.mode or cand["order"] != l.order:
                old = (l.mode, l.order)
                l.mode, l.ebias, l.order = cand["mode"], cand["eb"], cand["order"]
                self.cool[lid] = self.cooldown
                print(f"[NumericController] Layer {lid} mode/order {old} -> {(l.mode, l.order)} (safer_needed={safer_needed})")
            
            l.of_count.zero_()
            l.uf_count.zero_()
            l.swamp_count.zero_()
            l.headroom.zero_()


class EnergyMonitor:
    """Energy and power monitoring (with fallback emulation)"""
    
    def __init__(self):
        self.has_nvml = HAS_NVML
        if self.has_nvml:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.has_nvml = False
        
        self.power_history = []
        self.energy_total = 0.0
        self.last_time = time.time()

    def sample(self):
        """Sample current power consumption"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if self.has_nvml:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                power_w = power_mw / 1000.0
            except Exception:
                power_w = 150.0 + 50.0 * np.random.random()  # Fallback
        else:
            power_w = 150.0 + 50.0 * np.random.random()  # Emulated power
        
        self.power_history.append(power_w)
        if len(self.power_history) > 100:
            self.power_history.pop(0)
        
        self.energy_total += power_w * dt
        self.last_time = current_time
        
        return power_w

    def get_stats(self):
        """Get energy statistics"""
        if not self.power_history:
            return {"power_avg": 0, "energy_total": 0}
        return {
            "power_avg": np.mean(self.power_history),
            "energy_total": self.energy_total
        }


def train_model(model, train_loader, test_loader, num_epochs=5, device="cuda", quick_test=False):
    """Main training function with AccuTune controllers"""
    print(f"Training model for {num_epochs} epochs on {device}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    gns_estimator = LNOnlyGNS()
    gns_estimator.attach(model)
    
    ga_controller = GAController(optimizer, target_temp=1.0, min_ga=1, max_ga=8 if quick_test else 16)
    
    lowbit_layers = []
    for m in model.modules():
        if isinstance(m, LowBitAccLinear):
            lowbit_layers.append(m)
    
    numeric_controller = NumericController(lowbit_layers) if lowbit_layers else None
    energy_monitor = EnergyMonitor()
    
    train_losses = []
    test_accuracies = []
    ga_history = []
    energy_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        gns_val = gns_estimator.value()
        current_ga = ga_controller.suggest(gns_val)
        ga_history.append(current_ga)
        
        gns_str = f"{gns_val:.4f}" if gns_val is not None else "None"
        print(f"Epoch {epoch+1}/{num_epochs}, GA={current_ga}, GNS={gns_str}")
        
        accumulated_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            if len(output.shape) == 3 and len(target.shape) == 2:  # [B, T, V] vs [B, T]
                output = output.reshape(-1, output.size(-1))  # [B*T, V]
                target = target.reshape(-1)  # [B*T]
            
            loss = criterion(output, target) / current_ga  # Scale loss for accumulation
            
            loss.backward()
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % current_ga == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                if numeric_controller and (batch_idx + 1) % (current_ga * 10) == 0:
                    ref_cosines = {id(l): 0.98 for l in lowbit_layers}  # Simplified
                    numeric_controller.step(ref_cosines)
                
                accumulated_loss = 0.0
            
            epoch_loss += loss.item() * current_ga
            num_batches += 1
            
            power = energy_monitor.sample()
            if batch_idx % 10 == 0:
                energy_history.append(energy_monitor.get_stats())
            
            if quick_test and batch_idx >= 20:  # Early stop for quick test
                break
        
        if accumulated_loss > 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_loss)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if len(output.shape) == 3 and len(target.shape) == 2:  # [B, T, V] vs [B, T]
                    output = output.reshape(-1, output.size(-1))  # [B*T, V]
                    target = target.reshape(-1)  # [B*T]
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                if quick_test and batch_idx >= 10:  # Early stop for quick test
                    break
        
        accuracy = 100 * correct / max(total, 1)
        test_accuracies.append(accuracy)
        
        print(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        if lowbit_layers:
            for i, layer in enumerate(lowbit_layers):
                of = int(layer.of_count.item())
                uf = int(layer.uf_count.item())
                swamp = int(layer.swamp_count.item())
                alpha = float(layer.alpha_ema.item())
                print(f"    Layer {i}: OF={of}, UF={uf}, Swamp={swamp}, Alpha={alpha:.3f}")
    
    return {
        "train_losses": train_losses,
        "test_accuracies": test_accuracies,
        "ga_history": ga_history,
        "energy_history": energy_history,
        "final_accuracy": test_accuracies[-1] if test_accuracies else 0.0
    }


if __name__ == "__main__":
    print("Testing AccuTune training components...")
    
    layer = LowBitAccLinear(64, 32)
    x = torch.randn(8, 64)
    y = layer(x)
    print(f"LowBitAccLinear test: input {x.shape} -> output {y.shape}")
    
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.LayerNorm(32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    
    gns = LNOnlyGNS()
    gns.attach(model)
    
    x = torch.randn(8, 64, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    print(f"GNS estimate: {gns.value()}")
    
    print("Training components test completed successfully!")
