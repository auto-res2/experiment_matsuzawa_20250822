import os
import math
import time
import copy
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# ==========================
# Reproducibility & Utilities
# ==========================

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_device(device_str: Optional[str] = None) -> torch.device:
    if device_str is not None:
        if device_str.startswith('cuda') and torch.cuda.is_available():
            return torch.device(device_str)
        if device_str == 'cpu':
            return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================
# Differentiable STE Fake-Quant helpers
# ==========================

def ste_round(x):
    # Straight-through estimator for rounding
    return (x.round() - x).detach() + x


def fake_quantize_ste(x: torch.Tensor, clip: torch.Tensor, bits: Optional[int]) -> torch.Tensor:
    # Symmetric uniform quantization with STE; clip is positive scalar/tensor
    if bits is None:
        return x
    qmax = 2 ** (bits - 1) - 1
    qmin = -2 ** (bits - 1)
    s = clip / qmax
    s = torch.clamp(s, min=1e-8)
    x_clamped = torch.clamp(x, -clip, clip)
    q = ste_round(x_clamped / s)
    q = torch.clamp(q, qmin, qmax)
    y = q * s
    return y


def per_channel_fake_quantize_w_ste(w: torch.Tensor, s_mult: torch.Tensor, bits: Optional[int]) -> torch.Tensor:
    # Per-output-channel symmetric quantization with STE and learnable scale multiplier
    if bits is None:
        return w
    out_c = w.shape[0]
    w_flat = w.view(out_c, -1)
    base = w_flat.abs().amax(dim=1) / (2 ** (bits - 1) - 1 + 1e-8)
    eff_scale = base * s_mult.abs()
    eff_scale = torch.clamp(eff_scale, min=1e-8)
    y = []
    for c in range(out_c):
        wc = w_flat[c]
        sc = eff_scale[c]
        q = ste_round(wc / sc)
        q = torch.clamp(q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        yc = q * sc
        y.append(yc)
    y = torch.stack(y, dim=0).view_as(w)
    return y


# ==========================
# Quantization-capable toy layers/model
# ==========================

class QAct(nn.Module):
    def __init__(self, init_clip: float = 1.0):
        super().__init__()
        self.act_clip = nn.Parameter(torch.tensor(float(init_clip)))

    def forward(self, x: torch.Tensor, bits: Optional[int] = None) -> torch.Tensor:
        clip = self.act_clip.abs()
        return fake_quantize_ste(x, clip, bits)


class QConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.s_mult = nn.Parameter(torch.ones(out_ch))  # per-channel weight scale multiplier
        self.act = QAct(init_clip=1.0)

    def forward(self, x: torch.Tensor, bits: Optional[int] = None) -> torch.Tensor:
        w_q = per_channel_fake_quantize_w_ste(self.conv.weight, self.s_mult, bits)
        b = self.conv.bias
        x = F.conv2d(x, w_q, b, stride=self.conv.stride, padding=self.conv.padding)
        x = self.bn(x)
        x = F.relu(x)
        x = self.act(x, bits)
        return x


class QLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias)
        self.s_mult = nn.Parameter(torch.ones(out_f))  # per-output weight scale multiplier

    def forward(self, x: torch.Tensor, bits: Optional[int] = None) -> torch.Tensor:
        w = self.fc.weight
        w_q = per_channel_fake_quantize_w_ste(w, self.s_mult, bits)
        b = self.fc.bias
        return F.linear(x, w_q, b)


class QToyCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=10):
        super().__init__()
        self.c1 = QConv2d(in_ch, 16, 3, padding=1)
        self.c2 = QConv2d(16, 32, 3, stride=2, padding=1)
        self.c3 = QConv2d(32, 64, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = QLinear(64, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, bits: Optional[int] = None) -> torch.Tensor:
        x = self.c1(x, bits)
        x = self.c2(x, bits)
        x = self.c3(x, bits)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x, bits)
        return x

    def forward_fp(self, x):
        return self.forward(x, bits=None)

    def forward_fakequant(self, x, bits=8):
        return self.forward(x, bits=bits)

    def forward_quant(self, x):
        return self.forward(x, bits=8)


# ==========================
# Int8 Row-Quantized Sketch maps
# ==========================

class Int8RowQuant:
    def __init__(self, W_fp32: torch.Tensor, eps: float = 1e-8):
        W = W_fp32.detach().cpu().float()
        s = W.abs().amax(dim=1, keepdim=True) / 127.0 + eps
        Q = torch.clamp((W / s).round(), -127, 127).to(torch.int8)
        self.Q = Q  # CPU by default, moved on demand in matvec
        self.scale = s.squeeze(1).contiguous()  # per-row scale (float32)

    @classmethod
    def from_q_and_scale(cls, Q: torch.Tensor, scale: torch.Tensor):
        obj = cls.__new__(cls)
        obj.Q = Q.to(torch.int8).contiguous()
        obj.scale = scale.float().contiguous()
        return obj

    def matvec(self, x_fp16: torch.Tensor) -> torch.Tensor:
        # Compute y = (Q * scale) @ x, moving buffers to x's device as needed
        device = x_fp16.device
        x = x_fp16.detach().float().to(device)
        y_int = torch.matmul(self.Q.float().to(device), x)
        return y_int * self.scale.to(device)


# ==========================
# Gate & Adapters
# ==========================

class GateMLP(nn.Module):
    def __init__(self, in_dim: int, K: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, K)
        )
        self.K = K

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(feats), dim=-1)


class MomentumAdapter:
    def __init__(self, params_ref: List[torch.Tensor], lr: float = 0.1, mom: float = 0.9, clip: float = 0.1):
        self.ref = params_ref
        self.lr, self.mom, self.clip = lr, mom, clip
        self.v = [torch.zeros_like(p) for p in self.ref]

    def step(self, delta_flat: torch.Tensor):
        offset = 0
        for i, p in enumerate(self.ref):
            n = p.numel()
            if n == 0:
                continue
            d = delta_flat[offset:offset + n].view_as(p)
            offset += n
            d = torch.clamp(d, -self.clip, self.clip)
            self.v[i].mul_(self.mom).add_((1 - self.mom) * d)
            with torch.no_grad():
                p.add_(-self.lr * self.v[i])


class BiasNormAdapter(MomentumAdapter):
    pass


class QuantScaleCalibrator(MomentumAdapter):
    pass


# (Spectral adapter placeholder for future iterations)
class SpectralDiagonalAdapter:
    def __init__(self):
        pass


# ==========================
# SketchBank core
# ==========================

class SketchBank:
    def __init__(self, P_fp32: torch.Tensor, S_groups: Dict[str, List[Int8RowQuant]], gate: GateMLP, ema_beta: float = 0.99):
        self.P = P_fp32.contiguous()
        self.S_groups = S_groups
        self.gate = gate
        self.ema_beta = ema_beta
        self.ema_norm = 1.0

    def project_error(self, e: torch.Tensor) -> torch.Tensor:
        e_prime = self.P.to(e.device).t() @ e
        with torch.no_grad():
            self.ema_norm = self.ema_beta * self.ema_norm + (1 - self.ema_beta) * (e_prime.norm().item() + 1e-8)
        return e_prime

    def apply_update(self, adapters: Dict[str, MomentumAdapter], e_prime: torch.Tensor, gate_feats: torch.Tensor):
        w = self.gate(gate_feats)
        for g, S_list in self.S_groups.items():
            upd = 0.0
            for k, Sk in enumerate(S_list):
                yk = Sk.matvec(e_prime.half())
                upd = upd + w[k] * yk
            scale = min(1.0, 1.0 / (1e-6 + self.ema_norm))
            adapters[g].step(scale * upd)


# ==========================
# Gate features & Parity guard
# ==========================

def compute_gate_feats(model: nn.Module, x: torch.Tensor, logits: torch.Tensor, e_prime: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        p = torch.softmax(logits.flatten(), dim=-1)
        entropy = -(p * torch.clamp(p, 1e-6, 1.0).log()).sum().unsqueeze(0)
        top2 = torch.topk(p, k=2).values
        margin = (top2[0] - top2[1]).unsqueeze(0)
        x_energy = x.float().pow(2).mean().unsqueeze(0)
        e_norm = torch.tensor([e_prime.norm().item()], device=logits.device)
        feats = torch.cat([entropy, margin, x_energy.to(logits.device), e_norm])  # 4-D feature
        return feats


def parity_guard(model: QToyCNN, x: torch.Tensor, bits_list=[8, 6, 4], tau: float = 0.02) -> Tuple[bool, float]:
    with torch.no_grad():
        y_fp = model.forward_fp(x)
        bad = False
        worst = 0.0
        for b in bits_list:
            y_q = model.forward_fakequant(x, bits=b)
            gap = (y_fp - y_q).abs().mean().item()
            worst = max(worst, gap)
            if gap > tau:
                bad = True
        return (not bad), worst


# ==========================
# Offline precomputation: learn P and fit S_g via ridge regression
# ==========================

def learn_P(model: QToyCNN, data: List[Tuple[torch.Tensor, int]], C: int, r_e: int = 8, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    model.eval().to(device)
    E = []
    with torch.no_grad():
        for xb, y in data:
            xb = xb.to(device)
            logits = model.forward_fp(xb)
            p = torch.softmax(logits.flatten(), dim=-1)
            e = p - F.one_hot(torch.tensor(y, device=device), C).float()
            E.append(e.unsqueeze(0).cpu())
    E = torch.cat(E, dim=0)  # N x C
    Cov = (E.t() @ E) / (max(1, E.size(0) - 1))
    U, S, Vt = torch.linalg.svd(Cov, full_matrices=False)
    P = U[:, :r_e].contiguous()  # C x r_e
    return P


def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
    vecs = []
    for p in params:
        if p.numel() == 0:
            continue
        vecs.append(p.reshape(-1))
    if len(vecs) == 0:
        return torch.zeros(0)
    return torch.cat(vecs)


def jacobian_T_e(model: QToyCNN, xb: torch.Tensor, e_vec: torch.Tensor, params: List[torch.Tensor], mode: str = 'fp') -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    if mode == 'quant':
        ylog = model.forward_fakequant(xb, bits=8)
    else:
        ylog = model.forward_fp(xb)
    grads = torch.autograd.grad(ylog, params, grad_outputs=e_vec.unsqueeze(0), retain_graph=False, allow_unused=True)
    gflat = []
    for g in grads:
        if g is None:
            gflat.append(torch.zeros(0, device=xb.device))
        else:
            gflat.append(g.reshape(-1))
    if len(gflat) == 0:
        return torch.zeros(0, device=xb.device)
    return torch.cat(gflat)


def fit_S_group(model: QToyCNN, data: List[Tuple[torch.Tensor, int]], P: torch.Tensor,
                 group_params: List[torch.Tensor], lam: float = 1e-3, mode: str = 'fp', device: torch.device = torch.device('cpu')) -> Int8RowQuant:
    model.eval().to(device)
    Zs, Ts = [], []
    C = model.num_classes
    for xb, y in data:
        xb = xb.to(device)
        with torch.no_grad():
            logits = model.forward_fp(xb)
            p = torch.softmax(logits.flatten(), dim=-1)
            e = p - F.one_hot(torch.tensor(y, device=device), C).float()
            e_prime = (P.to(device).t() @ e).detach().cpu()
            Zs.append(e_prime.unsqueeze(0))
        t = jacobian_T_e(model, xb, e.to(device), group_params, mode=mode).detach().cpu()
        Ts.append(t.unsqueeze(0))
    Z = torch.cat(Zs, dim=0)  # N x r_e
    T = torch.cat(Ts, dim=0)  # N x p
    if T.numel() == 0 or T.shape[1] == 0:
        S = torch.zeros((0, Z.shape[1]), dtype=torch.float32)
        return Int8RowQuant(S)
    ZtZ = Z.t() @ Z
    A = ZtZ + lam * torch.eye(ZtZ.size(0))
    S = torch.linalg.solve(A, Z.t() @ T)  # r_e x p
    S = S.t().contiguous()  # p x r_e
    return Int8RowQuant(S)


# ==========================
# Collect parameters for adapters (toy model)
# ==========================

def collect_bias_bn_params_toy(model: QToyCNN) -> List[torch.Tensor]:
    ps = []
    for mod in [model.c1, model.c2, model.c3]:
        if mod.conv.bias is not None:
            ps.append(mod.conv.bias)
        ps.append(mod.bn.weight)
        ps.append(mod.bn.bias)
    if model.head.fc.bias is not None:
        ps.append(model.head.fc.bias)
    return ps


def collect_quant_params_toy(model: QToyCNN) -> List[torch.Tensor]:
    ps = []
    for mod in [model.c1, model.c2, model.c3]:
        ps.append(mod.s_mult)
        ps.append(mod.act.act_clip)
    ps.append(model.head.s_mult)
    return ps


# ==========================
# Synthetic data generators (multiple patterns)
# ==========================

def make_synthetic_image(sample_id: int, C: int, pattern: str, img_size: int = 32) -> Tuple[torch.Tensor, int]:
    rng = np.random.RandomState(sample_id)
    x = np.zeros((3, img_size, img_size), dtype=np.float32)
    label = rng.randint(0, C)
    color = (label / max(C - 1, 1))
    x[:, img_size // 4: 3 * img_size // 4, img_size // 4: 3 * img_size // 4] = color
    if pattern == 'bright':
        x = np.clip(x + 0.25, 0.0, 1.0)
    elif pattern == 'contrast':
        x = np.clip((x - 0.5) * 1.5 + 0.5, 0.0, 1.0)
    elif pattern == 'noise':
        x = np.clip(x + rng.normal(scale=0.1, size=x.shape).astype(np.float32), 0.0, 1.0)
    elif pattern == 'patch':
        i = rng.randint(0, img_size - 4)
        j = rng.randint(0, img_size - 4)
        x[:, i:i + 4, j:j + 4] = 1.0
    return torch.from_numpy(x).unsqueeze(0), int(label)


def build_synthetic_dataset(N: int, C: int, patterns: List[str]) -> List[Tuple[torch.Tensor, int]]:
    data = []
    for i in range(N):
        pat = patterns[i % len(patterns)]
        xb, y = make_synthetic_image(i, C, pat)
        data.append((xb, y))
    return data


def make_synthetic_logmel(sample_id: int, C: int, pattern: str, H: int = 32, W: int = 32) -> Tuple[torch.Tensor, int]:
    rng = np.random.RandomState(sample_id)
    x = rng.normal(loc=0.0, scale=0.3, size=(1, H, W)).astype(np.float32)
    label = rng.randint(0, C)
    band = (label * W) // C
    x[:, :, band:band + max(1, W // (2 * C))] += 1.0
    if pattern == 'noise_low_snr':
        x += rng.normal(scale=0.5, size=x.shape).astype(np.float32)
    elif pattern == 'reverb':
        kernel = np.ones((1, 1, 1, 5), dtype=np.float32) / 5.0
        x = torch.from_numpy(x).unsqueeze(0)
        k = torch.from_numpy(kernel)
        x = F.conv2d(x, k, padding=(0, 2)).squeeze(0).numpy()
    x = np.clip(x, -2.0, 2.0)
    return torch.from_numpy(x).unsqueeze(0), int(label)


def build_synthetic_kws_dataset(N: int, C: int, patterns: List[str]) -> List[Tuple[torch.Tensor, int]]:
    data = []
    for i in range(N):
        pat = patterns[i % len(patterns)]
        xb, y = make_synthetic_logmel(i, C, pat)
        data.append((xb, y))
    return data


# ==========================
# Training utilities
# ==========================

def pretrain_toy_model(model: QToyCNN, data: List[Tuple[torch.Tensor, int]], steps: int = 200, lr: float = 1e-3, device: torch.device = torch.device('cpu')):
    model.train().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for t in range(steps):
        xb, y = data[t % len(data)]
        xb = xb.to(device)
        y = torch.tensor([y], device=device)
        opt.zero_grad(set_to_none=True)
        logits = model.forward_fp(xb)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def eval_accuracy(model: QToyCNN, data: List[Tuple[torch.Tensor, int]], mode: str = 'fp', device: torch.device = torch.device('cpu')) -> float:
    model.eval().to(device)
    correct = 0
    with torch.no_grad():
        for xb, y in data:
            xb = xb.to(device)
            if mode == 'fp':
                logits = model.forward_fp(xb)
            else:
                logits = model.forward_quant(xb)
            pred = logits.argmax(dim=-1).item()
            correct += int(pred == y)
    return correct / max(1, len(data))


def confusion_matrix(model: QToyCNN, data: List[Tuple[torch.Tensor, int]], C: int, mode: str = 'fp', device: torch.device = torch.device('cpu')) -> np.ndarray:
    cm = np.zeros((C, C), dtype=np.int64)
    model.eval().to(device)
    with torch.no_grad():
        for xb, y in data:
            xb = xb.to(device)
            if mode == 'fp':
                logits = model.forward_fp(xb)
            else:
                logits = model.forward_quant(xb)
            pred = logits.argmax(dim=-1).item()
            cm[y, pred] += 1
    return cm


def plot_curves(xs, curves: Dict[str, List[float]], title: str, ylabel: str, filename: str):
    plt.figure(figsize=(5, 3))
    for k, ys in curves.items():
        plt.plot(xs, ys, label=k)
    plt.title(title)
    plt.xlabel('step')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_confusion(cm: np.ndarray, classes: List[str], title: str, filename: str):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# ==========================
# Pretrain entrypoints
# ==========================

def pretrain_vision(config: Dict):
    device = get_device(config.get('device'))
    seed = int(config.get('seed', 1234))
    set_seed(seed)

    models_dir = config.get('models_dir', 'models')
    ensure_dir(models_dir)

    C = int(config.get('num_classes', 10))
    steps = int(config.get('training', {}).get('steps', 120))
    lr = float(config.get('training', {}).get('lr', 5e-3))

    # Data
    train_data = build_synthetic_dataset(N=400, C=C, patterns=['clean'])
    test_clean = build_synthetic_dataset(N=100, C=C, patterns=['clean'])

    # Model
    model = QToyCNN(in_ch=3, num_classes=C).to(device)

    # Pretrain
    losses = pretrain_toy_model(model, train_data, steps=steps, lr=lr, device=device)

    # Eval
    base_fp_acc = eval_accuracy(model, test_clean, mode='fp', device=device)
    base_q_acc = eval_accuracy(model, test_clean, mode='quant', device=device)

    # Save
    save_path = os.path.join(models_dir, 'qtcnn_vision.pt')
    torch.save({'state_dict': model.state_dict(), 'config': config, 'seed': seed}, save_path)

    return {
        'model': model,
        'train_data': train_data,
        'test_clean': test_clean,
        'losses': losses,
        'base_fp_acc': base_fp_acc,
        'base_q_acc': base_q_acc,
        'save_path': save_path
    }


def pretrain_kws(config: Dict):
    device = get_device(config.get('device'))
    seed = int(config.get('seed', 1234))
    set_seed(seed)

    models_dir = config.get('models_dir', 'models')
    ensure_dir(models_dir)

    C = int(config.get('kws_num_classes', 12))
    steps = int(config.get('training', {}).get('steps', 120))
    lr = float(config.get('training', {}).get('lr', 5e-3))

    train_data = build_synthetic_kws_dataset(N=400, C=C, patterns=['clean'])
    test_clean = build_synthetic_kws_dataset(N=120, C=C, patterns=['clean'])

    model = QToyCNN(in_ch=1, num_classes=C).to(device)

    losses = pretrain_toy_model(model, train_data, steps=steps, lr=lr, device=device)

    base_fp_acc = eval_accuracy(model, test_clean, mode='fp', device=device)
    base_q_acc = eval_accuracy(model, test_clean, mode='quant', device=device)

    save_path = os.path.join(models_dir, 'qtcnn_kws.pt')
    torch.save({'state_dict': model.state_dict(), 'config': config, 'seed': seed}, save_path)

    return {
        'model': model,
        'train_data': train_data,
        'test_clean': test_clean,
        'losses': losses,
        'base_fp_acc': base_fp_acc,
        'base_q_acc': base_q_acc,
        'save_path': save_path
    }


# ==========================
# CLI
# ==========================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SketchBank - Pretraining script (toy)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML')
    parser.add_argument('--task', type=str, default='vision', choices=['vision', 'kws'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    ensure_dir(cfg.get('models_dir', 'models'))
    ensure_dir(cfg.get('data_dir', 'data'))

    if args.task == 'vision':
        out = pretrain_vision(cfg)
        print(f"[Pretrain-Vision] Saved model to {out['save_path']}")
        print(f"[Pretrain-Vision] Clean FP acc={out['base_fp_acc']:.3f}, 8-bit acc={out['base_q_acc']:.3f}")
    else:
        out = pretrain_kws(cfg)
        print(f"[Pretrain-KWS] Saved model to {out['save_path']}")
        print(f"[Pretrain-KWS] Clean FP acc={out['base_fp_acc']:.3f}, 8-bit acc={out['base_q_acc']:.3f}")
