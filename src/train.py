"""
BEMeGA Training Module
Implements the episodic adapter and training logic for few-shot learning
"""

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression


@dataclass
class DictionaryBank:
    subspace_atoms: List[torch.Tensor]
    diag_bins: List[List[int]]
    mean_norm: float


def build_random_dictionary(D: int, top_k: int = 32, n_atoms: int = 5, n_bins: int = 8, device: str = "cpu") -> DictionaryBank:
    """Build a random dictionary bank with orthonormal subspaces and diagonal bins"""
    atoms = []
    for _ in range(n_atoms):
        A = torch.randn(D, top_k, device=device)
        Q, _ = torch.linalg.qr(A, mode="reduced")
        atoms.append(Q)
    
    all_dims = list(range(D))
    bins = []
    stride = max(1, D // n_bins)
    for i in range(n_bins):
        start = i * stride
        end = D if i == n_bins - 1 else (i + 1) * stride
        bins.append(all_dims[start:end])
    
    return DictionaryBank(subspace_atoms=atoms, diag_bins=bins, mean_norm=1.0)


def within_between_trace_ratio(Z: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute within/between class scatter ratio"""
    classes = torch.unique(y)
    mu_all = Z.mean(dim=0, keepdim=True)
    tw = torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
    tb = torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
    
    for c in classes:
        Xc = Z[y == c]
        mc = Xc.mean(dim=0, keepdim=True)
        Xc0 = Xc - mc
        tw = tw + torch.sum(Xc0.pow(2))
        diff = (mc - mu_all)
        tb = tb + Xc.shape[0] * torch.sum(diff.pow(2))
    
    return tw / (tb + eps)


def small_sample_silhouette_torch(Z: torch.Tensor, y: torch.Tensor) -> float:
    """Fast silhouette proxy with cosine distance on normalized features"""
    if torch.unique(y).numel() < 2:
        return 0.0
    
    X = F.normalize(Z, dim=1)
    S = X @ X.t()
    Dcos = 1.0 - S
    y_np = y.detach().cpu().numpy()
    n = Z.size(0)
    
    a = torch.zeros(n, device=Z.device, dtype=Z.dtype)
    b = torch.full((n,), float("inf"), device=Z.device, dtype=Z.dtype)
    
    for idx in range(n):
        c = y_np[idx]
        mask_same = torch.tensor(y_np == c, device=Z.device)
        mask_diff = torch.tensor(y_np != c, device=Z.device)
        
        same_idx = torch.where(mask_same)[0]
        if same_idx.numel() > 1:
            a[idx] = Dcos[idx, mask_same].sum() / (same_idx.numel() - 1)
        
        unique_classes = np.unique(y_np[mask_diff.detach().cpu().numpy()])
        if unique_classes.size == 0:
            b[idx] = 0.0
        else:
            bmin = float("inf")
            for oc in unique_classes:
                mask_oc = torch.tensor(y_np == oc, device=Z.device)
                mean_dist = Dcos[idx, mask_oc].mean()
                bmin = min(bmin, float(mean_dist.item()))
            b[idx] = bmin
    
    s = (b - a) / (torch.maximum(a, b) + 1e-6)
    return float(torch.nan_to_num(s, nan=0.0).mean().item())


def intra_class_nn_ratio_torch(Z: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    """Ratio of nearest neighbors that share the same label"""
    X = Z
    D = torch.cdist(X, X)
    D = D + torch.eye(D.size(0), device=D.device) * 1e6
    idx = torch.topk(D, k=k, largest=False).indices
    y_rep = y.view(-1, 1).repeat(1, k)
    hits = (y_rep == y[idx]).sum().item()
    total = X.size(0) * k
    return hits / max(1, total)


def james_stein_trace_shrink_torch(per_class: List[torch.Tensor]) -> torch.Tensor:
    """Shrink per-class trace to pooled trace when sample is tiny"""
    D = per_class[0].size(1)
    pooled = torch.cat(per_class, dim=0)
    pooled_tr = pooled.var(dim=0, unbiased=True).sum()
    traces = []
    
    for X in per_class:
        n = X.size(0)
        if n > 1:
            raw_tr = X.var(dim=0, unbiased=True).sum()
        else:
            raw_tr = pooled_tr
        alpha = min(1.0, max(0.0, (D - 2) / max(1, n - 1)))
        tr = (1 - alpha) * raw_tr + alpha * pooled_tr
        traces.append(tr)
    
    return torch.stack(traces)


def compute_support_stats(Z: torch.Tensor, y: torch.Tensor, mean_norm_ref: float = 1.0) -> Dict:
    """Compute comprehensive support statistics for the adapter"""
    classes = torch.unique(y)
    per_class = [Z[y == c] for c in classes]
    means = torch.stack([X.mean(0) for X in per_class], dim=0)
    pairwise = torch.cdist(means, means)
    norm_var = (Z.pow(2).sum(1)).var(unbiased=False)
    
    small = any([X.size(0) < 5 for X in per_class])
    if small:
        traces_t = james_stein_trace_shrink_torch(per_class)
    else:
        traces_t = torch.stack([X.var(dim=0, unbiased=True).sum() for X in per_class])
    
    sil = small_sample_silhouette_torch(Z, y)
    nnr = intra_class_nn_ratio_torch(Z, y, k=1)
    tw_tb = within_between_trace_ratio(Z, y)
    mean_norm = torch.norm(Z, dim=1).mean().item()
    norm_scale_ratio = mean_norm / (mean_norm_ref + 1e-9)
    trace_disp = traces_t.std(unbiased=False)
    
    return {
        "means": means,
        "pairwise": pairwise,
        "norm_var": norm_var,
        "traces": traces_t,
        "silhouette": sil,
        "intra_nn_ratio": nnr,
        "tw_tb_ratio": tw_tb,
        "norm_scale_ratio": norm_scale_ratio,
        "trace_dispersion": trace_disp,
        "N": int(len(classes)),
        "k": int(per_class[0].size(0)) if len(per_class) > 0 else 0,
    }


def build_P(weights: torch.Tensor, atoms: List[torch.Tensor], d: int) -> torch.Tensor:
    """Build projection matrix from dictionary atoms"""
    D = atoms[0].size(0)
    Ucols = sum([A.size(1) for A in atoms])
    U = torch.zeros(D, Ucols, device=atoms[0].device, dtype=atoms[0].dtype)
    col = 0
    for w, A in zip(weights, atoms):
        k_i = A.size(1)
        U[:, col:col + k_i] = w * A
        col += k_i
    Q, _ = torch.linalg.qr(U, mode="reduced")
    d_sel = min(d, Q.size(1))
    return Q[:, :d_sel]


def stiefel_penalty(P: torch.Tensor) -> torch.Tensor:
    """Stiefel manifold penalty to keep projection orthonormal"""
    I = torch.eye(P.size(1), device=P.device, dtype=P.dtype)
    return torch.norm(P.T @ P - I, p="fro") ** 2


@dataclass
class CovParams:
    sigma2: Dict[int, torch.Tensor]
    B_dict: Dict[int, torch.Tensor]


def spherical_kmeans_torch(X: torch.Tensor, m: int, iters: int = 5) -> torch.Tensor:
    """Simple spherical k-means for prototype multiplicity"""
    Xn = F.normalize(X, dim=1)
    n, d = Xn.shape
    if n < m:
        mean = Xn.mean(0, keepdim=True)
        return mean.repeat(m, 1)
    
    centers = [Xn[torch.randint(0, n, (1,)).item()]]
    for _ in range(1, m):
        sims = torch.stack([Xn @ c for c in centers], dim=1)
        dists = 1 - sims.max(dim=1).values
        idx = torch.argmax(dists)
        centers.append(Xn[idx])
    
    C = torch.stack(centers, dim=0)
    for _ in range(iters):
        sims = Xn @ C.T
        assign = sims.argmax(dim=1)
        newC = []
        for j in range(m):
            mask = (assign == j)
            if mask.any():
                cj = Xn[mask].mean(0)
                cj = F.normalize(cj, dim=0)
            else:
                cj = C[j]
            newC.append(cj)
        C = torch.stack(newC, dim=0)
    
    avg_norm = X.norm(dim=1).mean().item()
    return C * avg_norm


def mahalanobis_mixture_logits(Zq: torch.Tensor, protos: Dict[int, List[torch.Tensor]], 
                              covs: CovParams, priors: Dict[int, float]) -> torch.Tensor:
    """Compute logits using Mahalanobis distance with mixture of prototypes"""
    classes = sorted(list(protos.keys()))
    C = len(classes)
    logits_per_c = []
    
    for c in classes:
        centers = protos[c]
        sigma2 = covs.sigma2[c]
        B = covs.B_dict[c]
        
        if B.numel() == 0:
            inv = 1.0 / sigma2
            dist_min = torch.full((Zq.size(0),), float('inf'), device=Zq.device, dtype=Zq.dtype)
            for mu in centers:
                diff = Zq - mu.unsqueeze(0)
                dval = diff.pow(2).sum(1) * inv
                dist_min = torch.minimum(dist_min, dval)
        else:
            BtB = (B.T @ B) / sigma2
            M = torch.linalg.inv(torch.eye(BtB.size(0), device=B.device, dtype=B.dtype) + BtB)
            dist_min = torch.full((Zq.size(0),), float('inf'), device=Zq.device, dtype=Zq.dtype)
            for mu in centers:
                diff = Zq - mu.unsqueeze(0)
                t1 = diff / sigma2
                t2 = (t1 @ B) @ M @ (B.T / sigma2)
                Qx = t1 - t2
                dval = (diff * Qx).sum(1)
                dist_min = torch.minimum(dist_min, dval)
        
        log_prior = math.log(priors.get(c, 1.0 / C))
        logits_per_c.append((-dist_min + log_prior).unsqueeze(1))
    
    return torch.cat(logits_per_c, dim=1)


@dataclass
class AdapterOutputs:
    P: torch.Tensor
    tau_P: float
    d: int
    r: int
    m_vec: Dict[int, int]
    cov_params: CovParams
    priors: Dict[int, float]
    risk_hat: float
    lam: float


class BEMeGAAdapter(nn.Module):
    """BEMeGA episodic adapter for support-conditioned geometry selection"""
    
    def __init__(self, D: int, dict_bank: DictionaryBank, max_d: int = 32, max_r: int = 6, device: str = "cpu"):
        super().__init__()
        self.D = D
        self.db = dict_bank
        self.max_d = min(max_d, D)
        self.max_r = min(max_r, self.max_d)
        self.device = device
        
        proxy_dim = 6
        self.atom_w = nn.Linear(proxy_dim, len(self.db.subspace_atoms))
        self.scale_head = nn.Linear(proxy_dim, 3)
        self.risk_head = nn.Linear(proxy_dim, 2)
        
        for m in [self.atom_w, self.scale_head, self.risk_head]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, stats: Dict) -> AdapterOutputs:
        with torch.no_grad():
            pvec = torch.tensor([
                float(stats["norm_var"]) if torch.is_tensor(stats["norm_var"]) else float(stats["norm_var"]),
                float(stats["tw_tb_ratio"]) if torch.is_tensor(stats["tw_tb_ratio"]) else float(stats["tw_tb_ratio"]),
                float(stats["trace_dispersion"]) if torch.is_tensor(stats["trace_dispersion"]) else float(stats["trace_dispersion"]),
                float(stats["silhouette"]),
                float(stats["intra_nn_ratio"]),
                float(stats["norm_scale_ratio"])],
                device=self.device, dtype=torch.float32)
        
        w = F.softmax(self.atom_w(pvec), dim=0)
        scales = self.scale_head(pvec)
        tau_P = 0.5 + 2.0 * torch.sigmoid(scales[0])
        d = max(1, min(self.max_d, int(self.max_d * torch.sigmoid(scales[1]))))
        r = max(0, min(self.max_r, int(self.max_r * torch.sigmoid(scales[2]))))
        
        P = build_P(w, self.db.subspace_atoms, d)
        
        risk_out = self.risk_head(pvec)
        risk_hat = torch.sigmoid(risk_out[0]).item()
        lam = 0.1 + 0.8 * torch.sigmoid(risk_out[1]).item()
        
        N = stats["N"]
        classes = list(range(N))
        
        m_vec = {}
        sigma2_dict = {}
        B_dict = {}
        priors = {}
        
        for c in classes:
            k = stats["k"]
            if k <= 2:
                m_vec[c] = 1
            elif stats["silhouette"] > 0.3 and stats["intra_nn_ratio"] < 0.7:
                m_vec[c] = min(3, max(1, k // 2))
            else:
                m_vec[c] = 1
            
            sigma2_dict[c] = torch.tensor(1.0, device=self.device, dtype=torch.float32)
            if r > 0:
                B_dict[c] = torch.randn(d, r, device=self.device, dtype=torch.float32) * 0.1
            else:
                B_dict[c] = torch.empty(d, 0, device=self.device, dtype=torch.float32)
            
            priors[c] = 1.0 / N
        
        cov_params = CovParams(sigma2=sigma2_dict, B_dict=B_dict)
        
        return AdapterOutputs(
            P=P, tau_P=float(tau_P), d=d, r=r, m_vec=m_vec,
            cov_params=cov_params, priors=priors, risk_hat=risk_hat, lam=lam
        )


class ProtoNetBaseline(nn.Module):
    """Simple ProtoNet baseline for comparison"""
    
    def __init__(self, D: int, device: str = "cpu"):
        super().__init__()
        self.D = D
        self.device = device
    
    def forward(self, support_data: torch.Tensor, support_labels: torch.Tensor, 
                query_data: torch.Tensor) -> torch.Tensor:
        classes = torch.unique(support_labels)
        prototypes = []
        
        for c in classes:
            class_samples = support_data[support_labels == c]
            prototype = class_samples.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes, dim=0)
        distances = torch.cdist(query_data, prototypes)
        logits = -distances
        
        return logits


def train_bemega_adapter(adapter: BEMeGAAdapter, episodes: List, 
                        num_epochs: int = 100, lr: float = 1e-3, device: str = "cpu") -> Dict:
    """Train the BEMeGA adapter on episodes"""
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for support_data, support_labels, query_data, query_labels in episodes:
            support_data = support_data.to(device)
            support_labels = support_labels.to(device)
            query_data = query_data.to(device)
            query_labels = query_labels.to(device)
            
            stats = compute_support_stats(support_data, support_labels)
            adapter_out = adapter(stats)
            
            Zs_proj = support_data @ adapter_out.P * adapter_out.tau_P
            Zq_proj = query_data @ adapter_out.P * adapter_out.tau_P
            
            classes = torch.unique(support_labels).tolist()
            protos = {}
            
            for c in classes:
                class_data = Zs_proj[support_labels == c]
                if adapter_out.m_vec[c] == 1:
                    protos[c] = [class_data.mean(0)]
                else:
                    centers = spherical_kmeans_torch(class_data, adapter_out.m_vec[c])
                    protos[c] = [centers[i] for i in range(centers.size(0))]
            
            logits = mahalanobis_mixture_logits(Zq_proj, protos, adapter_out.cov_params, adapter_out.priors)
            
            loss = F.cross_entropy(logits, query_labels)
            
            bound_penalty = 0.0
            if hasattr(stats, 'norm_var'):
                bound_penalty += 0.01 * stats["norm_var"]
            if hasattr(stats, 'tw_tb_ratio'):
                bound_penalty += 0.01 * stats["tw_tb_ratio"]
            if hasattr(stats, 'trace_dispersion'):
                bound_penalty += 0.01 * stats["trace_dispersion"]
            
            total_loss = loss + bound_penalty + 0.001 * stiefel_penalty(adapter_out.P)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            pred = logits.argmax(dim=1)
            epoch_acc += (pred == query_labels).float().mean().item()
        
        avg_loss = epoch_loss / len(episodes)
        avg_acc = epoch_acc / len(episodes)
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.4f}")
    
    return {"losses": losses, "accuracies": accuracies}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing training module on {device}")
    
    D = 128
    dict_bank = build_random_dictionary(D, device=device)
    adapter = BEMeGAAdapter(D, dict_bank, device=device)
    
    from preprocess import SyntheticEpisodeGenerator, create_episode_batch
    generator = SyntheticEpisodeGenerator(D=D, device=device)
    episodes = create_episode_batch(generator, "standard", N=5, k=5, q=15, batch_size=20)
    
    print("Training BEMeGA adapter...")
    results = train_bemega_adapter(adapter, episodes, num_epochs=50, device=device)
    print(f"Final accuracy: {results['accuracies'][-1]:.4f}")
    print("Training module test completed successfully!")
