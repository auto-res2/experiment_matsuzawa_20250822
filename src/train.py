import os
import math
import time
import random
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.fft as tfft

from einops import rearrange
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.signal.windows import dpss as scipy_dpss
except Exception:
    scipy_dpss = None

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix
except Exception:
    PCA = None
    KMeans = None
    confusion_matrix = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def save_pdf(fig, filename):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def sample_positions(L: int, rate: float, min_start: int) -> torch.Tensor:
    n = max(1, int(L * rate))
    if min_start >= L:
        min_start = max(0, L - 1)
    idx = torch.tensor(sorted(random.sample(range(min_start, L), k=n)))
    return idx


class TensorSketcher:
    def __init__(self, V: int, r: int, k: int, device: torch.device, antithetic: bool = True):
        self.V, self.r, self.k = V, r, k
        self.device = device
        self.antithetic = antithetic
        self.rehash()

    def rehash(self):
        self.hash = torch.randint(0, self.r, (self.k, self.V), device=self.device)
        self.sign = (torch.randint(0, 2, (self.k, self.V), device=self.device).float() * 2.0 - 1.0)
        if self.antithetic:
            self.hash2 = self.hash.clone()
            self.sign2 = -self.sign
        else:
            self.hash2 = torch.zeros_like(self.hash)
            self.sign2 = torch.zeros_like(self.sign)

    def onehot_sketch(self, x: torch.Tensor, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = torch.zeros(B, self.r, device=self.device)
        idx = h.gather(0, x)
        val = s.gather(0, x)
        out.scatter_add_(1, idx.unsqueeze(1), val.unsqueeze(1))
        return out

    def degree_k_sketch(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        sk = 1
        for i in range(self.k):
            y = self.onehot_sketch(tokens[i], self.hash[i], self.sign[i])
            yf = tfft.rfft(y, n=2*self.r, dim=1)
            sk = yf if isinstance(sk, int) else (sk * yf)
        out = tfft.irfft(sk, n=2*self.r, dim=1)[..., :self.r]
        if self.antithetic:
            sk2 = 1
            for i in range(self.k):
                y2 = self.onehot_sketch(tokens[i], self.hash2[i], self.sign2[i])
                yf2 = tfft.rfft(y2, n=2*self.r, dim=1)
                sk2 = yf2 if isinstance(sk2, int) else (sk2 * yf2)
            out2 = tfft.irfft(sk2, n=2*self.r, dim=1)[..., :self.r]
            out = 0.5 * (out + out2)
        return out


class ORFFeatures(nn.Module):
    def __init__(self, in_dim: int, r: int, sigma: float, device: torch.device):
        super().__init__()
        assert r % 2 == 0
        W = torch.randn(in_dim, r, device=device)
        q, _ = torch.linalg.qr(W, mode='reduced')
        b = 2*math.pi*torch.rand(r, device=device)
        self.register_buffer('W', q)
        self.register_buffer('b', b)
        self.sigma = float(sigma) if sigma > 1e-8 else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = (x @ self.W) / self.sigma + self.b
        return torch.cos(proj) * (2.0 / self.W.shape[1])**0.5


class PSDTracker:
    def __init__(self, L0: int = 512, K: int = 3, ema: float = 0.95, device: torch.device = torch.device('cpu'), pca_components: int = 12, late_lowfreq_boost: float = 2.0):
        self.L0, self.K, self.ema = L0, K, ema
        self.device = device
        self.late_lowfreq_boost = late_lowfreq_boost
        self.R = pca_components
        self.registered = False
        self.step = 0
        if scipy_dpss is not None:
            NW = 2.5
            tap, _ = scipy_dpss(L0, NW, K, sym=False, return_ratios=True)
            self.tapers = torch.from_numpy(np.array(tap).copy()).to(self.device)
        else:
            n = torch.arange(L0, device=self.device)
            w = 0.5 - 0.5*torch.cos(2*math.pi*n/(L0-1))
            self.tapers = torch.stack([w for _ in range(K)], dim=0)

    def _register(self, n_bands: int):
        self.psd_ema = torch.zeros(n_bands, device=self.device)
        self.registered = True

    def compute_psd_batch(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L0 = x.shape
        if L0 != self.L0:
            if L0 < self.L0:
                pad_size = self.L0 - L0
                x = torch.nn.functional.pad(x, (0, pad_size), mode='constant', value=0)
            else:
                x = x[:, :, :self.L0]
            L0 = self.L0
        K = self.tapers.shape[0]
        Xk = []
        for k in range(K):
            tapered = x * self.tapers[k]
            fk = tfft.rfft(tapered, dim=-1)
            Pk = (fk.abs() ** 2)
            Xk.append(Pk)
        P = torch.stack(Xk, dim=0)
        if K >= 2:
            P_jk = (P.sum(0, keepdim=True) - P) / (K - 1)
            P_mean = P_jk.mean(0)
        else:
            P_mean = P.mean(0)
        BC, F = B*C, P_mean.shape[-1]
        Z = P_mean.reshape(BC, F)
        if PCA is not None and F > 1:
            try:
                pca = PCA(n_components=min(self.R, F))
                Zp = pca.fit_transform(Z.detach().cpu().numpy())
                Zr = torch.from_numpy(pca.inverse_transform(Zp)).to(self.device)
                P_denoised = Zr.reshape(B, C, F).mean(dim=(0,1))
            except Exception:
                P_denoised = Z.mean(dim=(0,1))
        else:
            P_denoised = Z.mean(dim=0)
        if not self.registered:
            self._register(P_denoised.numel())
        self.psd_ema = self.ema * self.psd_ema + (1 - self.ema) * P_denoised
        self.step += 1
        return self.psd_ema

    def band_weights(self, F: int, late_stage: bool) -> torch.Tensor:
        w = torch.ones(F, device=self.device)
        if late_stage:
            f = torch.linspace(1, F, F, device=self.device)
            w = 1.0 / f.sqrt()
            w = (w / w.mean()) * self.late_lowfreq_boost
        return w


class HankelSketcher:
    def __init__(self, L0: int = 512, bandwidth: int = 64, r: int = 128, device: torch.device = torch.device('cpu'), huber_delta: float = 1.0):
        self.L0, self.bw, self.r = L0, bandwidth, r
        self.device = device
        self.delta = huber_delta
        self.G = torch.randn(self.r, self.bw, device=self.device) / (self.r ** 0.5)

    @staticmethod
    def autocorr_1d(x: torch.Tensor, maxlag: int) -> torch.Tensor:
        x = x - x.mean()
        r = F.conv1d(x.view(1,1,-1), x.view(1,1,-1).flip(-1))
        mid = r.shape[-1] // 2
        r = r.view(-1)
        out = r[mid:mid+maxlag+1]
        return out

    def hankel_block(self, r: torch.Tensor) -> torch.Tensor:
        n = self.bw
        if r.numel() < n:
            pad_size = n - r.numel()
            r = torch.nn.functional.pad(r, (0, pad_size), mode='constant', value=0)
        if r.numel() < n:
            return torch.zeros(n, n, device=r.device)
        unfolded = r.unfold(0, n, 1)
        if unfolded.size(0) < n:
            pad_rows = n - unfolded.size(0)
            padding = torch.zeros(pad_rows, n, device=r.device)
            unfolded = torch.cat([unfolded, padding], dim=0)
        return unfolded[:n]

    def huber(self, diff: torch.Tensor) -> torch.Tensor:
        absd = diff.abs()
        quad = 0.5 * absd.pow(2)
        lin = self.delta * (absd - 0.5*self.delta)
        return torch.where(absd <= self.delta, quad, lin).sum()

    def sketch_loss(self, r_model: torch.Tensor, r_data: torch.Tensor) -> torch.Tensor:
        Hm = self.hankel_block(r_model)
        Hd = self.hankel_block(r_data)
        Hm = self.G @ Hm
        Hd = self.G @ Hd
        return self.huber(Hm - Hd)


class ToySSMLayer(nn.Module):
    def __init__(self, d_model: int, n_modes: int = 8, kernel_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.kernel_len = kernel_len
        self.log_alpha = nn.Parameter(torch.randn(n_modes) * 0.1)
        self.log_decay = nn.Parameter(torch.randn(n_modes) - 3.0)
        self.omega = nn.Parameter(torch.rand(n_modes) * math.pi)
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def build_kernel(self, L: Optional[int] = None) -> torch.Tensor:
        L = L or self.kernel_len
        t = torch.arange(L, device=self.log_alpha.device).float()
        alpha = F.softplus(self.log_alpha)
        decay = torch.exp(self.log_decay).clamp(max=0.999)
        omega = self.omega
        terms = []
        for i in range(self.n_modes):
            terms.append(alpha[i] * (decay[i] ** t) * torch.cos(omega[i] * t))
        k = torch.stack(terms, dim=0).sum(0)
        return k

    def get_kernel(self, L: int) -> torch.Tensor:
        return self.build_kernel(L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        h = self.in_proj(x)
        k = self.build_kernel(self.kernel_len)
        w = k.view(1, 1, -1).repeat(d, 1, 1)
        y = F.conv1d(h.transpose(1,2), w, padding=self.kernel_len-1, groups=d)
        y = y.transpose(1,2)[:, :L, :]
        y = self.out_proj(y)
        return self.norm(x + y)


class TinySSMModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4, n_modes: int = 8, kernel_len: int = 64, num_classes: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([ToySSMLayer(d_model, n_modes, kernel_len) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor, return_hidden: bool = False):
        h = self.embed(tokens)
        for lyr in self.layers:
            h = lyr(h)
        h = self.norm(h)
        h_cls = h.mean(dim=1)
        logits = self.cls_head(h_cls)
        if return_hidden:
            return logits, h, [lyr for lyr in self.layers]
        return logits

    def get_ssm_layers(self) -> List[ToySSMLayer]:
        return list(self.layers)


class SMTPHead(nn.Module):
    def __init__(self, d_model: int, r: int):
        super().__init__()
        self.proj = nn.Linear(d_model, r)
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


def impulse_response_layer(layer: ToySSMLayer, L0: int) -> torch.Tensor:
    return layer.get_kernel(L0)


def spectral_kernel_matching_loss(k_model: torch.Tensor, psd_emp: torch.Tensor, late_stage: bool = False, tracker: Optional[PSDTracker] = None) -> torch.Tensor:
    F = psd_emp.shape[0]
    Kf = torch.fft.rfft(k_model, n=psd_emp.numel()*2)
    Pm = (Kf.abs() ** 2)[:F]
    Pm = Pm / (Pm.mean() + 1e-8)
    Pd = psd_emp / (psd_emp.mean() + 1e-8)
    if tracker is not None:
        w = tracker.band_weights(F, late_stage)
    else:
        w = torch.ones(F, device=psd_emp.device)
    return ((w * (Pm - Pd)) ** 2).mean()


def spectral_diversity_loss(layer_psds: List[torch.Tensor]) -> torch.Tensor:
    n = len(layer_psds)
    if n < 2:
        return torch.tensor(0.0, device=layer_psds[0].device)
    S = torch.stack(layer_psds, dim=0)
    S = S / (S.norm(dim=1, keepdim=True) + 1e-8)
    G = S @ S.T
    offdiag = G - torch.diag(torch.diag(G))
    return offdiag.pow(2).mean()


def parseval_tie_loss(k_model: torch.Tensor, x_batch_psd: torch.Tensor) -> torch.Tensor:
    Kf = torch.fft.rfft(k_model)
    Pm = (Kf.abs() ** 2) / (Kf.numel())
    Pd = x_batch_psd / (x_batch_psd.mean() + 1e-8)
    Pm = Pm[:Pd.numel()] / (Pm.mean() + 1e-8)
    return ((Pm - Pd) ** 2).mean()


def stability_barrier(params: List[torch.nn.Parameter], margin: float = 0.02) -> torch.Tensor:
    if not params:
        return torch.tensor(0.0)
    dev = params[0].device
    loss = torch.tensor(0.0, device=dev)
    for p in params:
        loss = loss + torch.relu(margin - torch.abs(p)).mean()
    return loss


class IDSPTplusplus:
    def __init__(self, psd_tracker: PSDTracker, hankel_sketcher: HankelSketcher, device: torch.device):
        self.psd_tracker = psd_tracker
        self.hankel = hankel_sketcher
        self.device = device

    def compute_losses(self, model: TinySSMModel, batch: Dict[str, torch.Tensor], step: int, total_steps: int, 
                      lambda1: float = 0.5, lambda2: float = 0.1, lambda3: float = 0.05) -> Dict[str, torch.Tensor]:
        
        tokens = batch['tokens']
        B, L = tokens.shape
        
        logits, hidden, ssm_layers = model(tokens, return_hidden=True)
        
        spt_loss = F.cross_entropy(logits, batch['labels'])
        
        late_stage = step > 0.8 * total_steps
        
        smtp_loss = torch.tensor(0.0, device=self.device)
        skm_loss = torch.tensor(0.0, device=self.device)
        hsm_loss = torch.tensor(0.0, device=self.device)
        
        if step > 0.1 * total_steps:
            x_windows = hidden[:, :self.psd_tracker.L0, :].transpose(1, 2)
            psd_emp = self.psd_tracker.compute_psd_batch(x_windows)
            
            layer_psds = []
            for layer in ssm_layers:
                k = impulse_response_layer(layer, self.psd_tracker.L0)
                layer_psd = torch.fft.rfft(k).abs() ** 2
                layer_psds.append(layer_psd[:len(psd_emp)])
                skm_loss += spectral_kernel_matching_loss(k, psd_emp, late_stage, self.psd_tracker)
            
            skm_loss /= len(ssm_layers)
            
            if len(layer_psds) > 1:
                diversity_loss = spectral_diversity_loss(layer_psds)
                skm_loss += 0.02 * diversity_loss
        
        if step > 0.2 * total_steps:
            for layer in ssm_layers:
                k = impulse_response_layer(layer, self.hankel.L0)
                r_model = self.hankel.autocorr_1d(k, self.hankel.bw)
                
                x_flat = hidden.mean(dim=0).mean(dim=1)
                r_data = self.hankel.autocorr_1d(x_flat, self.hankel.bw)
                
                hsm_loss += self.hankel.sketch_loss(r_model, r_data)
            
            hsm_loss /= len(ssm_layers)
        
        stability_params = []
        for layer in ssm_layers:
            stability_params.extend([layer.log_decay, layer.omega])
        
        stability_loss = stability_barrier(stability_params)
        
        total_loss = spt_loss + lambda1 * smtp_loss + lambda2 * skm_loss + lambda3 * hsm_loss + 0.01 * stability_loss
        
        return {
            'total_loss': total_loss,
            'spt_loss': spt_loss,
            'smtp_loss': smtp_loss,
            'skm_loss': skm_loss,
            'hsm_loss': hsm_loss,
            'stability_loss': stability_loss
        }


def train_model(model: TinySSMModel, train_loader: DataLoader, val_loader: DataLoader, 
                device: torch.device, num_epochs: int = 10, lr: float = 1e-3) -> Dict:
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    psd_tracker = PSDTracker(L0=256, device=device)
    hankel_sketcher = HankelSketcher(L0=256, device=device)
    idspt = IDSPTplusplus(psd_tracker, hankel_sketcher, device)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    total_steps = len(train_loader) * num_epochs
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for step, batch in enumerate(pbar):
            batch = to_device(batch, device)
            assert isinstance(batch, dict)
            
            optimizer.zero_grad()
            
            global_step = epoch * len(train_loader) + step
            losses = idspt.compute_losses(model, batch, global_step, total_steps)
            
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += losses['total_loss'].item()
            
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'spt': f"{losses['spt_loss'].item():.4f}",
                'skm': f"{losses['skm_loss'].item():.4f}",
                'hsm': f"{losses['hsm_loss'].item():.4f}"
            })
        
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = to_device(batch, device)
                    assert isinstance(batch, dict)
                    logits = model(batch['tokens'])
                    loss = F.cross_entropy(logits, batch['labels'])
                    val_loss += loss.item()
                    
                    pred = logits.argmax(dim=1)
                    correct += (pred == batch['labels']).sum().item()
                    total += batch['labels'].size(0)
                except Exception as e:
                    print(f"Validation batch failed: {e}")
                    raise e
        
        if total > 0:
            val_acc = correct / total
        else:
            print(f"Warning: No validation samples processed. total={total}")
            val_acc = 0.0
        
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.4f}')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'model': model
    }
