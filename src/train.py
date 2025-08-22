import os
import time
import math
import argparse
import json
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

# -----------------------------
# Utilities and directories
# -----------------------------

def ensure_dirs():
    os.makedirs('.research/iteration1/images', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def set_seed(seed: int = 1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

# -----------------------------
# Graph utilities
# -----------------------------

def build_row_stochastic(edge_index: torch.Tensor, num_nodes: int) -> torch.sparse.FloatTensor:
    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    A = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
    A = A.coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
    invdeg = 1.0 / deg
    ii = torch.arange(num_nodes, dtype=torch.long)
    Dinv = torch.sparse_coo_tensor(torch.stack([ii, ii]), invdeg, (num_nodes, num_nodes))
    P = torch.sparse.mm(Dinv, A).coalesce()
    return P


def precompute_multi_hop(P: torch.sparse.FloatTensor, X: torch.Tensor, L: int) -> List[torch.Tensor]:
    H = [X]
    cur = X
    for _ in range(1, L + 1):
        cur = torch.sparse.mm(P, cur)
        H.append(cur)
    return H

# -----------------------------
# MiGAD core modules
# -----------------------------

class GroupHead(nn.Module):
    def __init__(self, in_dim: int, G: int, hidden: int = 64, top_rho: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, G)
        )
        self.top_rho = top_rho
        self.register_buffer('prototypes', torch.zeros(G, in_dim))
        self.use_proto = False

    @torch.no_grad()
    def init_prototypes(self, X: torch.Tensor):
        N = X.size(0)
        G = self.prototypes.size(0)
        idx = torch.randperm(N)[:G]
        self.prototypes.copy_(X[idx])
        self.use_proto = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(X)
        if self.use_proto:
            Xn = F.normalize(X, dim=-1)
            Pn = F.normalize(self.prototypes, dim=-1)
            sim = Xn @ Pn.t()
            logits = logits + 0.1 * sim
        S = F.softmax(logits, dim=-1)
        if self.top_rho is not None and self.top_rho < S.size(1):
            k = self.top_rho
            topk = torch.topk(S, k=k, dim=1)
            mask = torch.zeros_like(S)
            mask.scatter_(1, topk.indices, 1.0)
            S = S * mask
            S = S / (S.sum(dim=1, keepdim=True) + 1e-8)
        return S

class Compatibility(nn.Module):
    def __init__(self, G: int, rank: int = 8):
        super().__init__()
        self.U = nn.Parameter(torch.randn(G, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(G, rank) * 0.01)

    def forward(self) -> torch.Tensor:
        C_logits = self.U @ self.V.t()
        C = F.softmax(C_logits, dim=1)
        return C

@torch.no_grad()
def diffuse_groups(P: torch.sparse.FloatTensor, S: torch.Tensor, L: int) -> List[torch.Tensor]:
    M = [S]
    cur = S
    for _ in range(1, L + 1):
        cur = torch.sparse.mm(P, cur)
        M.append(cur)
    return M

@torch.no_grad()
def compute_base_alignment(S: torch.Tensor, C: torch.Tensor, M_list: List[torch.Tensor]) -> torch.Tensor:
    SC = S @ C
    A = []
    for Mk in M_list:
        a_k = (SC * Mk).sum(dim=1)
        A.append(a_k)
    A = torch.stack(A, dim=1).clamp(0.0, 1.0)
    return A

@torch.no_grad()
def compute_risk_from_alignment(A: torch.Tensor) -> torch.Tensor:
    return (1.0 - A).clamp(0.0, 1.0)

@torch.no_grad()
def compute_denoising_proxy(H_list: List[torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    N = H_list[0].size(0)
    L = len(H_list) - 1
    D = torch.zeros(N, L + 1, device=H_list[0].device)
    for k in range(1, L + 1):
        delta = (H_list[k] - H_list[k - 1])
        nrm = delta.norm(p=2, dim=1)
        mk = torch.median(nrm).clamp(min=1e-6)
        D[:, k] = torch.sigmoid(-(nrm / (mk * temperature)))
    return D

class BudgetedHopAttention(nn.Module):
    def __init__(self, L: int, d_in: int, d_out: int, shared_f: bool = True):
        super().__init__()
        self.L = L
        self.theta = nn.Parameter(torch.zeros(L + 1))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.tau = nn.Parameter(torch.tensor(0.5))
        if shared_f:
            self.f = nn.Linear(d_in, d_out)
            self.per_hop = False
        else:
            self.f_list = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(L + 1)])
            self.per_hop = True

    def forward(self,
                H_list: List[torch.Tensor],
                R: torch.Tensor,
                D: torch.Tensor,
                rho: torch.Tensor,
                hard_gate: Optional[torch.Tensor] = None,
                bisection_iters: int = 25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = H_list[0].size(0)
        Lp1 = self.L + 1
        s = self.theta.view(1, Lp1) - self.beta * R - self.gamma * D
        if hard_gate is not None:
            s = s.masked_fill(hard_gate, float('-inf'))
        tau = torch.clamp(self.tau, min=1e-3)
        lam_low = torch.zeros(N, device=s.device)
        lam_high = torch.full((N,), 100.0, device=s.device)

        def softmax_lambda(lam):
            logits = (s - lam.view(-1, 1) * R) / tau
            logits = logits - logits.max(dim=1, keepdim=True).values
            return torch.softmax(logits, dim=1)

        for _ in range(bisection_iters):
            lam_mid = 0.5 * (lam_low + lam_high)
            w = softmax_lambda(lam_mid)
            cons = (w * R).sum(dim=1)
            go_right = cons > rho
            lam_low = torch.where(go_right, lam_mid, lam_low)
            lam_high = torch.where(go_right, lam_high, lam_mid)
        lam = lam_high
        w = softmax_lambda(lam)

        if self.per_hop:
            Z = 0.0
            for k in range(Lp1):
                Z = Z + w[:, k:k+1] * self.f_list[k](H_list[k])
        else:
            Fk = [self.f(Hk) for Hk in H_list]
            Z = torch.stack(Fk, dim=1)
            Z = (w.unsqueeze(-1) * Z).sum(dim=1)
        return Z, w, lam

class MiGAD(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, G: int, L: int,
                 top_rho: int = 4, rank: int = 8, shared_f: bool = True,
                 risk_mode: str = 'align'):
        super().__init__()
        self.L = L
        self.group_head = GroupHead(in_dim, G, hidden, top_rho)
        self.compat = Compatibility(G, rank)
        self.hop_attn = BudgetedHopAttention(L, d_in=in_dim, d_out=hidden, shared_f=shared_f)
        self.pred = nn.Linear(hidden, out_dim)
        assert risk_mode in ['align']  # keep minimal fast default
        self.risk_mode = risk_mode

    def forward(self, P: torch.sparse.FloatTensor, X: torch.Tensor, rho: torch.Tensor,
                temp_d: float = 1.0, hard_kappa: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        H_list = precompute_multi_hop(P, X, self.L)
        S = self.group_head(X)
        C = self.compat()
        M_list = diffuse_groups(P, S, self.L)
        A = compute_base_alignment(S, C, M_list)
        R = compute_risk_from_alignment(A)
        D = compute_denoising_proxy(H_list, temperature=temp_d)
        hard_gate = (R > hard_kappa) if hard_kappa is not None else None
        Z, w, lam = self.hop_attn(H_list, R, D, rho, hard_gate)
        logits = self.pred(Z + X)
        aux = {'S': S, 'C': C, 'R': R, 'D': D, 'w': w, 'lambda': lam, 'H_list': H_list}
        return logits, aux

@torch.no_grad()
def build_budget(P: torch.sparse.FloatTensor, S: torch.Tensor, rho0: float, rho_min: float, rho_max: float, alpha: float = 0.7) -> torch.Tensor:
    maxS = S.max(dim=1).values
    b1 = 1.0 - maxS
    S1 = torch.sparse.mm(P, S)
    b2 = 1.0 - (S * S1).sum(dim=1)
    b = alpha * b1 + (1 - alpha) * b2
    b = (b - b.min()) / (b.max() - b.min() + 1e-6)
    rho = (rho0 * b).clamp(rho_min, rho_max)
    return rho

# -----------------------------
# Training helpers
# -----------------------------

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred[idx] == y[idx]).float().mean().item()


def plot_training_loss(losses: List[float], title: str, fname: str):
    plt.figure()
    plt.plot(losses, label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'[Figure saved] {fname}')


def train_migad(config: Dict):
    ensure_dirs()
    seed = int(config['seed'])
    set_seed(seed)

    data_path = config['data']['path']
    device_str = config['train'].get('device', 'auto')
    if device_str == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_str
    print(f"[Device] Using {device}")

    # Load data
    data = torch.load(data_path)
    X = data['X'].float()
    y = data['y'].long()
    edge_index = data['edge_index'].long()
    idx_train = data['idx_train'].long()
    idx_val = data['idx_val'].long()
    idx_test = data['idx_test'].long()

    N, d_in = X.size()
    num_classes = int(y.max().item()) + 1

    # Build P
    P = build_row_stochastic(edge_index, N)

    # Model
    L = int(config['model']['L'])
    G = int(config['model']['G'])
    hidden = int(config['model']['hidden'])
    top_rho = int(config['model']['top_rho'])
    rank = int(config['model'].get('rank', 8))

    model = MiGAD(d_in, hidden, num_classes, G, L, top_rho=top_rho, rank=rank, risk_mode='align').to(device)
    model.group_head.init_prototypes(X)

    # Move data
    X = X.to(device); y = y.to(device)
    P = P.coalesce().to(device)

    # Optim
    lr = float(config['train']['lr'])
    wd = float(config['train']['weight_decay'])
    epochs = int(config['train']['epochs'])
    warmup_epochs = int(config['train']['warmup_epochs'])
    eval_every = int(config['train']['eval_every'])
    lambda_mix = float(config['train']['lambda_mix'])
    rho0 = float(config['train']['rho0'])
    rho_min = float(config['train']['rho_min'])
    rho_max = float(config['train']['rho_max'])
    hard_kappa_warm = None
    hard_kappa_cool = float(config['train']['hard_kappa_cool'])
    patience = int(config['train'].get('patience', 30))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val = -1e9
    best = {}
    losses = []
    epochs_no_improve = 0

    print('[Training] Start MiGAD training')

    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        with torch.no_grad():
            S_now = model.group_head(X)
            rho = build_budget(P, S_now, rho0=rho0, rho_min=rho_min, rho_max=rho_max)
        temp_d = 1.0 if epoch < warmup_epochs else 0.6
        hard_kappa = hard_kappa_warm if epoch < warmup_epochs else hard_kappa_cool
        logits, aux = model(P, X, rho, temp_d=temp_d, hard_kappa=hard_kappa)
        loss_task = F.cross_entropy(logits[idx_train], y[idx_train])
        cons = (aux['w'] * aux['R']).sum(dim=1)
        viol = torch.clamp(cons - rho, min=0.0)
        mix_pen = viol.pow(2).mean()
        S = aux['S']; C = aux['C']
        ent_S = -(S.clamp_min(1e-8) * S.clamp_min(1e-8).log()).sum(dim=1).mean()
        kl_C_uniform = (C * (C.clamp_min(1e-8) * C.size(1)).log()).sum(dim=1).mean()
        coef_mix = (0.1 if epoch < warmup_epochs else lambda_mix)
        loss = loss_task + coef_mix * mix_pen + 1e-3 * ent_S + 1e-4 * kl_C_uniform
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        losses.append(loss.item())

        if epoch % eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits_eval, aux_eval = model(P, X, rho)
            acc_tr = accuracy_from_logits(logits_eval, y, idx_train)
            acc_v = accuracy_from_logits(logits_eval, y, idx_val)
            acc_te = accuracy_from_logits(logits_eval, y, idx_test)
            print(f"[Epoch {epoch:03d}] loss={loss.item():.4f} task={loss_task.item():.4f} mix_pen={mix_pen.item():.4f} | acc (tr/val/te) = {acc_tr:.3f}/{acc_v:.3f}/{acc_te:.3f}")
            if acc_v > best_val:
                best_val = acc_v
                best = {
                    'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'val': acc_v, 'test': acc_te, 'rho': rho.detach().cpu(), 'aux': {k: v.detach().cpu() for k, v in aux_eval.items()},
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[Early Stop] No improvement for {patience} eval steps. Stopping.")
                    break

    print(f"[Training] Best val={best.get('val', 0):.3f}, test={best.get('test', 0):.3f}")

    # Save model and artifacts
    model_path = os.path.join('models', config['output']['model_name'])
    torch.save({'config': config, 'model_state': best.get('state', model.state_dict()), 'best': best}, model_path)
    print(f"[Saved] Model checkpoint -> {model_path}")

    # Plot training loss
    img_dir = '.research/iteration1/images'
    ts = int(time.time())
    loss_fig = os.path.join(img_dir, f"training_loss_{ts}.pdf")
    plot_training_loss(losses, 'MiGAD Training Loss', loss_fig)

    # Save a brief metrics summary json for convenience
    summary = {
        'best_val_acc': float(best.get('val', 0.0)),
        'best_test_acc': float(best.get('test', 0.0)),
        'epochs_run': len(losses)
    }
    summary_path = os.path.join('models', f"{config['output']['model_name']}.summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Summary -> {summary_path}")

    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MiGAD model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to YAML config')
    args = parser.parse_args()

    ensure_dirs()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_migad(config)
