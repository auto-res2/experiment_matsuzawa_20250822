import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time


class Calibrator(nn.Module):
    """Identity-initialized elementwise calibrator before classifier."""
    
    def __init__(self, dim: int, ridge_lambda: float = 1e-3):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.ridge = ridge_lambda
        self.register_buffer("XtX", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("XtY", torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        return x * self.weight + self.bias

    @torch.no_grad()
    def update_ridge(self, x_feat: torch.Tensor, target: torch.Tensor, ema_m: float = 0.9):
        x2 = (x_feat ** 2).mean(dim=0)
        xy = (x_feat * target).mean(dim=0)
        self.XtX.mul_(ema_m).add_((1 - ema_m) * x2)
        self.XtY.mul_(ema_m).add_((1 - ema_m) * xy)
        denom = self.XtX + self.ridge
        self.weight.copy_(self.XtY / denom.clamp_min(1e-6))


class FeatureHeadWrapper(nn.Module):
    """Wrap a backbone to expose penultimate features and logits with optional calibrator."""
    
    def __init__(self, model: nn.Module, num_classes: int, add_calibrator: bool = True):
        super().__init__()
        self.backbone = model
        self.num_classes = num_classes
        
        if hasattr(model, 'fc'):
            feat_dim = model.fc.in_features
            self.classifier = nn.Linear(feat_dim, num_classes)
            with torch.no_grad():
                self.classifier.weight.copy_(model.fc.weight)
                self.classifier.bias.copy_(model.fc.bias)
            model.fc = nn.Identity()
        elif hasattr(model, 'classifier'):
            feat_dim = model.classifier.in_features
            self.classifier = nn.Linear(feat_dim, num_classes)
            with torch.no_grad():
                self.classifier.weight.copy_(model.classifier.weight)
                self.classifier.bias.copy_(model.classifier.bias)
            model.classifier = nn.Identity()
        else:
            raise ValueError("Model must have 'fc' or 'classifier' attribute")
            
        self.feat_dim = feat_dim
        self.calibrator = Calibrator(feat_dim) if add_calibrator else nn.Identity()

    def forward(self, x, return_features: bool = True):
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = torch.flatten(F.adaptive_avg_pool2d(feats, 1), 1)
        feats_cal = self.calibrator(feats)
        logits = self.classifier(feats_cal)
        if return_features:
            return feats_cal, logits
        return logits


class BNStatsHook:
    """Collects per-channel normalized activations for diagonal Fisher approximation."""
    
    def __init__(self, model: nn.Module):
        self.handles = []
        self.cache = {}
        self.modules = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                handle = m.register_forward_hook(self._hook(name))
                self.handles.append(handle)
                self.modules.append((name, m))

    def _hook(self, name):
        def fn(m, inp, out):
            x = inp[0].detach()
            dims = [0] + list(range(2, x.ndim))
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, unbiased=False, keepdim=True)
            xhat = (x - mean) / torch.sqrt(var + m.eps)
            ex2 = (xhat ** 2).mean(dim=dims).squeeze()
            self.cache[name] = ex2
        return fn

    def clear(self):
        self.cache.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class AETTALite:
    """Label-free accuracy estimator for trust region control."""
    
    def __init__(self, num_classes: int, N: int = 5, noise_std: float = 0.1):
        self.N = N
        self.noise_std = noise_std
        self.num_classes = num_classes
        self.ent_mu = 1.0
        self.ent_sigma = 1.0
        self.pdd_mu = 0.5
        self.pdd_sigma = 0.25
        self.m = 0.9

    @torch.no_grad()
    def estimate(self, feats: torch.Tensor, classifier: nn.Module) -> Dict[str, float]:
        logits_list = []
        for _ in range(self.N):
            noise = torch.randn_like(feats) * self.noise_std
            logits = classifier(feats + noise)
            logits_list.append(logits)
        
        logits_stk = torch.stack(logits_list, dim=0)
        p = F.softmax(logits_stk, dim=-1)
        mean_p = p.mean(dim=0)
        
        ent = -(mean_p * (mean_p.clamp_min(1e-8).log())).sum(dim=-1).mean().item()
        
        preds = p.argmax(dim=-1)
        disagree = 0.0
        total = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                disagree += (preds[i] != preds[j]).float().mean().item()
                total += 1
        pdd = disagree / max(total, 1)
        
        self.ent_mu = self.m * self.ent_mu + (1 - self.m) * ent
        self.ent_sigma = self.m * self.ent_sigma + (1 - self.m) * abs(ent - self.ent_mu)
        self.pdd_mu = self.m * self.pdd_mu + (1 - self.m) * pdd
        self.pdd_sigma = self.m * self.pdd_sigma + (1 - self.m) * abs(pdd - self.pdd_mu)
        
        z_ent = (ent - self.ent_mu) / (self.ent_sigma + 1e-6)
        z_pdd = (pdd - self.pdd_mu) / (self.pdd_sigma + 1e-6)
        s = -0.6 * z_ent - 0.6 * z_pdd
        a_hat = float(1.0 / (1.0 + math.exp(-s)))
        
        return {"acc_hat": a_hat, "entropy": float(ent), "pdd": float(pdd)}


@dataclass
class AnchorBuffer:
    """Buffer for low-entropy anchor samples."""
    max_size: int = 64
    feats: List[torch.Tensor] = field(default_factory=list)
    logits: List[torch.Tensor] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)

    def push(self, feats: torch.Tensor, logits: torch.Tensor):
        with torch.no_grad():
            p = F.softmax(logits, dim=-1)
            ent = (-(p * p.clamp_min(1e-8).log()).sum(dim=-1)).mean().item()
        
        if len(self.entropies) < self.max_size or ent < (np.percentile(self.entropies, 50) if self.entropies else float('inf')):
            self.feats.append(feats.detach().cpu())
            self.logits.append(logits.detach().cpu())
            self.entropies.append(ent)
            if len(self.feats) > self.max_size:
                idx = int(np.argmax(self.entropies))
                for arr in (self.feats, self.logits, self.entropies):
                    arr.pop(idx)

    def get_anchor_p(self, device, num_classes: int) -> Optional[torch.Tensor]:
        if not self.logits:
            return None
        logits = torch.cat(self.logits, dim=0).to(device)
        return F.softmax(logits, dim=-1).mean(dim=0)


class SnapOpt:
    """SNAP-TTA optimizer with forward-only Fisher preconditioning and trust region."""
    
    def __init__(self, model_wrap: FeatureHeadWrapper, trust_tau: float = 0.01, 
                 base_lr: float = 1e-3, topk_frac: float = 0.4, prox_lambda: float = 1e-3, 
                 fisher_eps: float = 1e-3, N_aetta: int = 5, grad_clip: Optional[float] = 5.0):
        self.mw = model_wrap
        self.trust_tau = trust_tau
        self.lr0 = base_lr
        self.topk = topk_frac
        self.prox = prox_lambda
        self.feps = fisher_eps
        self.grad_clip = grad_clip
        
        self.param_groups = []
        self.src_state = {}
        for n, m in self.mw.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.requires_grad_(True)
                for pname, p in m.named_parameters(recurse=False):
                    if pname in ("weight", "bias"):
                        self.param_groups.append((f"{n}.{pname}", p))
                        self.src_state[f"{n}.{pname}"] = p.detach().clone()
        
        if isinstance(self.mw.calibrator, Calibrator):
            for pname, p in self.mw.calibrator.named_parameters():
                self.param_groups.append((f"calib.{pname}", p))
                self.src_state[f"calib.{pname}"] = p.detach().clone()
        
        self.bn_hook = BNStatsHook(self.mw)
        self.F_diag: Dict[str, torch.Tensor] = {}
        self.F_ema_m = 0.9
        self.gA = None
        self.gA_m = 0.9
        self.anchor_buf = AnchorBuffer()
        self.grad_var = 0.0
        self.grad_var_m = 0.9
        self.aetta = AETTALite(num_classes=self.mw.num_classes, N=N_aetta, noise_std=0.1)

    def _compute_loss(self, x: torch.Tensor, xa_strong: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        feats, logits = self.mw(x, return_features=True)
        feats_s, logits_s = self.mw(xa_strong, return_features=True)
        
        p = F.softmax(logits, dim=-1)
        H_ent = -(p * p.clamp_min(1e-8).log()).sum(dim=-1).mean()
        
        anchor_p = self.anchor_buf.get_anchor_p(x.device, self.mw.num_classes)
        if anchor_p is not None:
            anchor_p = anchor_p.clamp_min(1e-8)
            p_clamp = p.clamp_min(1e-8)
            L_anchor = (anchor_p.unsqueeze(0) * (anchor_p.unsqueeze(0).log() - p_clamp.log())).sum(dim=-1).mean()
        else:
            L_anchor = torch.tensor(0.0, device=x.device)
        
        teacher_logits = logits.detach()
        t = F.softmax(teacher_logits, dim=-1)
        L_consist = -(t * F.log_softmax(logits_s, dim=-1)).sum(dim=-1).mean()
        
        w_e, w_a, w_c = 1.0, 0.1, 0.1
        
        total_loss = w_e * H_ent + w_a * L_anchor + w_c * L_consist
        
        loss_dict = {
            "entropy": H_ent.item(),
            "anchor": L_anchor.item() if isinstance(L_anchor, torch.Tensor) else L_anchor,
            "consistency": L_consist.item(),
            "total": total_loss.item()
        }
        
        return total_loss, loss_dict

    def _update_fisher(self, uncertainty: torch.Tensor):
        """Update diagonal Fisher approximation."""
        u_mean = uncertainty.mean().item()
        
        for name, p in self.param_groups:
            if "weight" in name and name.replace(".weight", "") in self.bn_hook.cache:
                bn_name = name.replace(".weight", "")
                ex2 = self.bn_hook.cache[bn_name]
                F_new = ex2 * u_mean
            elif "bias" in name:
                F_new = torch.full_like(p, u_mean)
            else:
                F_new = torch.ones_like(p) * u_mean
            
            if name in self.F_diag:
                self.F_diag[name] = self.F_ema_m * self.F_diag[name] + (1 - self.F_ema_m) * F_new
            else:
                self.F_diag[name] = F_new

    def step(self, x: torch.Tensor, xa_strong: torch.Tensor) -> Dict[str, float]:
        self.mw.train()
        
        loss, loss_dict = self._compute_loss(x, xa_strong)
        
        for name, p in self.param_groups:
            if p.grad is not None:
                p.grad.zero_()
        
        loss.backward()
        
        grads = {}
        for name, p in self.param_groups:
            if p.grad is not None:
                grads[name] = p.grad.clone()
            else:
                grads[name] = torch.zeros_like(p)
        
        feats, logits = self.mw(x, return_features=True)
        p = F.softmax(logits, dim=-1)
        uncertainty = 1 - (p ** 2).sum(dim=-1)
        self._update_fisher(uncertainty)
        
        gpre = {}
        for name, g in grads.items():
            Fi = self.F_diag.get(name, torch.ones_like(g))
            gpre[name] = g / (Fi + self.feps)
        
        if self.gA is not None:
            beta_cv = 0.1
            for name in gpre:
                if name in self.gA and self.gA[name].shape == gpre[name].shape:
                    gpre[name] = gpre[name] - beta_cv * self.gA[name]
        
        masked = {}
        for name, g in gpre.items():
            k = max(1, int(self.topk * g.numel()))
            vals = g.abs().flatten()
            if k < vals.numel():
                thr = torch.topk(vals, k)[0][-1]
                mask = (g.abs() >= thr).to(g.dtype)
            else:
                mask = torch.ones_like(g)
            masked[name] = g * mask
        
        feats, _ = self.mw(x, return_features=True)
        aetta_result = self.aetta.estimate(feats, self.mw.classifier)
        acc_hat = aetta_result["acc_hat"]
        
        grad_norm = sum(g.norm().item() for g in grads.values())
        self.grad_var = self.grad_var_m * self.grad_var + (1 - self.grad_var_m) * (grad_norm ** 2)
        lr = self.lr0 * (1 - acc_hat) / (math.sqrt(self.grad_var) + 1e-6)
        
        with torch.no_grad():
            for name, p in self.param_groups:
                if name in masked:
                    update = -lr * masked[name]
                    if self.grad_clip:
                        update = torch.clamp(update, -self.grad_clip, self.grad_clip)
                    p.add_(update)
        
        with torch.no_grad():
            for name, p in self.param_groups:
                src = self.src_state[name].to(p.device)
                p.add_(-self.prox * (p - src))
        
        feats, logits = self.mw(x, return_features=True)
        self.anchor_buf.push(feats, logits)
        
        if self.gA is None:
            self.gA = {name: g.clone() for name, g in gpre.items()}
        else:
            for name, g in gpre.items():
                if name in self.gA and self.gA[name].shape == g.shape:
                    self.gA[name] = self.gA_m * self.gA[name] + (1 - self.gA_m) * g
                else:
                    self.gA[name] = g.clone()
        
        self.bn_hook.clear()
        
        result = {
            "loss": loss_dict["total"],
            "entropy": loss_dict["entropy"],
            "anchor": loss_dict["anchor"],
            "consistency": loss_dict["consistency"],
            "acc_hat": acc_hat,
            "lr": lr,
            "grad_norm": grad_norm
        }
        
        return result


def create_model(num_classes: int = 10, pretrained: bool = True, add_calibrator: bool = True):
    """Create a ResNet18 model wrapped with FeatureHeadWrapper."""
    import torchvision.models as models
    
    if pretrained:
        model = models.resnet18(pretrained=True)
        if num_classes == 10:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
    else:
        model = models.resnet18(pretrained=False, num_classes=num_classes)
        if num_classes == 10:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
    
    wrapped_model = FeatureHeadWrapper(model, num_classes, add_calibrator=add_calibrator)
    
    return wrapped_model


class TentOptimizer:
    """Baseline Tent optimizer for comparison."""
    
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        self.model = model
        self.lr = lr
        
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.requires_grad_(True)
        
        bn_params = []
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                bn_params.extend(m.parameters())
        
        self.optimizer = torch.optim.Adam(bn_params, lr=lr)
    
    def step(self, x: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        
        if hasattr(self.model, 'forward') and 'return_features' in self.model.forward.__code__.co_varnames:
            _, logits = self.model(x, return_features=True)
        else:
            logits = self.model(x)
        
        p = F.softmax(logits, dim=-1)
        loss = -(p * p.clamp_min(1e-8).log()).sum(dim=-1).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item(), "entropy": loss.item()}


if __name__ == "__main__":
    print("Testing training components...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_model(num_classes=10, pretrained=False, add_calibrator=True)
    model = model.to(device)
    
    snap_opt = SnapOpt(model, base_lr=1e-3, trust_tau=0.01)
    
    batch_size = 32
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    xa_strong = torch.randn(batch_size, 3, 32, 32).to(device)
    
    result = snap_opt.step(x, xa_strong)
    print(f"Step result: {result}")
    
    print("Training components test completed successfully!")
