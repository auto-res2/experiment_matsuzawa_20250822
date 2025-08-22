import os
import json
import math
import time
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# -----------------------
# Paths and helpers
# -----------------------
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGES_DIR = os.path.join(PROJECT_DIR, '.research', 'iteration1', 'images')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'config')

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Core Toy Q-SHIFT Components
# -----------------------
class ToyVLM(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_classes=3, seed=42):
        super().__init__()
        set_seed(seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.txt_proj = nn.Linear(d_model, d_model)
        self.img_proj = nn.Linear(d_model, d_model)
        self.txt_q = nn.Linear(d_model, d_model)
        self.txt_k = nn.Linear(d_model, d_model)
        self.img_q = nn.Linear(d_model, d_model)
        self.img_k = nn.Linear(d_model, d_model)
        self.shallow_txt_1 = nn.Linear(d_model, d_model)
        self.shallow_txt_2 = nn.Linear(d_model, d_model)
        self.shallow_img_1 = nn.Linear(d_model, d_model)
        self.shallow_img_2 = nn.Linear(d_model, d_model)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, n_classes)
        )
        self.bit2noise = {8: 0.005, 6: 0.02, 4: 0.05}

    def _fake_tokens(self, text_len: int, img_len: int, decisive_txt: set, decisive_img: set, pattern: str):
        txt = torch.randn(text_len, self.d_model)
        img = torch.randn(img_len, self.d_model)
        subspace = torch.randn(self.d_model, 2)
        for i in decisive_txt:
            txt[i] += (subspace @ torch.tensor([1.0, 0.5])).to(txt)
        for j in decisive_img:
            img[j] += (subspace @ torch.tensor([1.0, 0.5])).to(img)
        if pattern == 'noisy':
            txt += 0.5 * torch.randn_like(txt)
            img += 0.5 * torch.randn_like(img)
        elif pattern == 'aligned':
            for i in decisive_txt:
                if len(decisive_img) > 0:
                    jj = random.choice(list(decisive_img))
                    txt[i] = 0.7 * txt[i] + 0.3 * img[jj]
        elif pattern == 'sparse_signal':
            for i in decisive_txt:
                txt[i] *= 1.3
            for j in decisive_img:
                img[j] *= 1.3
        return txt, img

    def _multihead_scaled_dot(self, Q, K, n_heads):
        Tq, D = Q.shape
        Tk = K.shape[0]
        head_dim = D // n_heads
        Qh = Q.view(Tq, n_heads, head_dim).transpose(0, 1)
        Kh = K.view(Tk, n_heads, head_dim).transpose(0, 1)
        attn = torch.einsum('htd,hkd->htk', Qh, Kh) / math.sqrt(head_dim)
        attn = F.softmax(attn, dim=-1)
        return attn

    def warm_pass(self, text_len, img_len, decisive_txt, decisive_img, pattern):
        txt, img = self._fake_tokens(text_len, img_len, decisive_txt, decisive_img, pattern)
        txt = self.txt_proj(txt)
        img = self.img_proj(img)
        A_l_txt = F.relu(self.shallow_txt_1(txt))
        A_lp1_txt = F.relu(self.shallow_txt_2(A_l_txt))
        A_l_img = F.relu(self.shallow_img_1(img))
        A_lp1_img = F.relu(self.shallow_img_2(A_l_img))
        attn_t2i = self._multihead_scaled_dot(self.txt_q(txt), self.img_k(img), self.n_heads)
        attn_i2t = self._multihead_scaled_dot(self.img_q(img), self.txt_k(txt), self.n_heads)
        return {
            'txt': txt, 'img': img,
            'A_l_txt': A_l_txt, 'A_lp1_txt': A_lp1_txt,
            'A_l_img': A_l_img, 'A_lp1_img': A_lp1_img,
            'attn_t2i': attn_t2i, 'attn_i2t': attn_i2t
        }

    def _conditional_entropy(self, A_l, A_lp1, n_bins=16):
        x = A_l.detach().flatten()
        y = A_lp1.detach().flatten()
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        y = (y - y.min()) / (y.max() - y.min() + 1e-6)
        xb = torch.clamp((x * (n_bins - 1)).long(), 0, n_bins - 1)
        yb = torch.clamp((y * (n_bins - 1)).long(), 0, n_bins - 1)
        H = torch.zeros(n_bins, n_bins)
        idx = xb * n_bins + yb
        H = H.view(-1)
        H.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        H = H.view(n_bins, n_bins)
        H = H / (H.sum() + 1e-6)
        Px = H.sum(dim=1, keepdim=True)
        Pyx = H / (Px + 1e-6)
        Hcond = -(H * (Pyx.add(1e-6).log())).sum().item()
        return Hcond

    def _sharedness_js(self, attn_t2i, attn_i2t):
        p = attn_t2i.mean(dim=0)
        q = attn_i2t.transpose(1, 2).mean(dim=0)
        m = 0.5 * (p + q) + 1e-6
        js = 0.5 * ((p * (p.add(1e-6).log() - m.log())).sum(dim=-1) +
                    (q * (q.add(1e-6).log() - m.log())).sum(dim=-1))
        s_txt = 1.0 - (js - js.min()) / (js.max() - js.min() + 1e-6)
        s_img = 1.0 - (js.mean())
        return s_txt, s_img

    def _apply_precision_noise(self, x, bits):
        noise = self.bit2noise.get(bits, 0.01)
        return x + noise * torch.randn_like(x)

    def forward_classify(self, sample: Dict, plan: Dict):
        out = self.warm_pass(sample['text_len'], sample['img_len'],
                             set(sample['decisive_indices'][0]), set(sample['decisive_indices'][1]), sample['pattern'])
        txt = out['txt']
        img = out['img']
        attn_t2i = out['attn_t2i']
        attn_i2t = out['attn_i2t']
        s_txt, _ = self._sharedness_js(attn_t2i, attn_i2t)
        k = max(1, int(len(s_txt) * float(plan.get('keep_ratio', 1.0))))
        top_idx = torch.topk(s_txt, k=k).indices
        txt_kept = txt[top_idx]
        depth_t = plan.get('depth_text', 1.0)
        depth_v = plan.get('depth_vision', 1.0)
        n_ref_t = 1 if depth_t <= 0.25 else 2 if depth_t <= 0.5 else 3 if depth_t <= 0.75 else 4
        n_ref_v = 1 if depth_v <= 0.25 else 2 if depth_v <= 0.5 else 3 if depth_v <= 0.75 else 4
        for _ in range(n_ref_t):
            txt_kept = F.relu(self.txt_proj(txt_kept))
        img_ref = img
        for _ in range(n_ref_v):
            img_ref = F.relu(self.img_proj(img_ref))
        bits = int(plan.get('precision_id', 8))
        txt_kept = self._apply_precision_noise(txt_kept, bits)
        img_ref = self._apply_precision_noise(img_ref, bits)
        t_vec = txt_kept.mean(dim=0)
        a = attn_t2i.mean(dim=0)[top_idx]
        img_weights = a.mean(dim=0)
        img_top = torch.topk(img_weights, k=min(len(img_weights), max(1, int(len(img_weights) * plan.get('keep_ratio', 1.0))))).indices
        i_vec = img_ref[img_top].mean(dim=0)
        fused = torch.cat([t_vec, i_vec], dim=-1)
        logits = self.cls_head(fused)
        return logits, {
            's_txt': s_txt.detach(),
            'attn_t2i': attn_t2i.detach(),
            'attn_i2t': attn_i2t.detach(),
            'A_l_txt': out['A_l_txt'].detach(), 'A_lp1_txt': out['A_lp1_txt'].detach(),
            'A_l_img': out['A_l_img'].detach(), 'A_lp1_img': out['A_lp1_img'].detach(),
        }

    def decode_with_kv(self, sample: Dict, plan: Dict, gen_len=64, safety=None):
        text_len = sample['text_len']
        kv_bits = int(plan.get('kv_bits', 8))
        head_drop = float(plan.get('head_drop', 0.0))
        evict_step = int(plan.get('evict_step', 0))
        keep_ratio = float(plan.get('keep_ratio', 1.0))
        precision_id = int(plan.get('precision_id', 8))
        depth_text = float(plan.get('depth_text', 1.0))
        depth_vision = float(plan.get('depth_vision', 1.0))
        bit_scale = {8: 1.0, 6: 0.85, 4: 0.7}[kv_bits]
        px_scale = {8: 1.0, 6: 0.9, 4: 0.8}[precision_id]
        lat_per_token = []
        kv_mem = []
        quality = 0.0
        backoff_triggers = 0
        kept_heads = int(self.n_heads * (1.0 - head_drop))
        kept_heads = max(1, kept_heads)
        current_seq = text_len
        base_margin = 0.8 * px_scale * (0.5 + 0.5 * depth_text) * (0.5 + 0.5 * depth_vision)
        for _ in range(gen_len):
            d_head = self.d_model // self.n_heads
            seq_after_evict = max(text_len, current_seq - (current_seq - evict_step if (evict_step > 0 and current_seq > evict_step) else 0))
            mem_bytes = seq_after_evict * kept_heads * d_head * (kv_bits / 8.0) * 2
            kv_mem.append(mem_bytes / 1e6)
            lat = (current_seq * kept_heads * bit_scale) * (1.5 - 0.5 * keep_ratio)
            lat_per_token.append(lat)
            margin = base_margin - 0.1 * (8 - kv_bits) / 4.0 - 0.1 * (8 - precision_id) / 4.0
            margin -= 0.2 * sample['difficulty']
            if safety is not None and (margin < safety['margin_thr']):
                kv_bits = 8
                precision_id = 8
                keep_ratio = min(1.0, keep_ratio + 0.25)
                kept_heads = self.n_heads
                backoff_triggers += 1
                base_margin = 0.85
            quality += max(0.0, margin)
            current_seq += 1
        return lat_per_token, kv_mem, quality / gen_len, backoff_triggers


class TorchOracle(nn.Module):
    def __init__(self, in_dim=11, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, feat_dict: Dict):
        with torch.no_grad():
            x = torch.tensor([build_oracle_feature(feat_dict)], dtype=torch.float32)
            y = self.forward(x).item()
            return max(0.1, y)


def build_oracle_feature(features: Dict) -> List[float]:
    return [
        float(features.get('text_len', 32)),
        float(features.get('img_len', 64)),
        float(features.get('keep_ratio', 1.0)),
        float(features.get('merge_size', 4)),
        float(features.get('depth_text', 1.0)),
        float(features.get('depth_vision', 1.0)),
        float(features.get('precision_id', 8)),
        float(features.get('kv_bits', 8)),
        float(features.get('head_drop', 0.0)),
        float(features.get('evict_step', 0)),
        float(features.get('batch_size', 1)),
    ]


def synth_latency_fn(features: Dict) -> float:
    t = features
    base = 0.2
    text_cost = 0.01 * t['text_len'] * (0.5 + 0.5 * t['depth_text']) * (t['precision_id'] / 8.0)
    img_cost = 0.008 * t['img_len'] * t['keep_ratio'] * (0.5 + 0.5 * t['depth_vision']) * (t['precision_id'] / 8.0)
    kv_cost = 0.003 * max(1, t['evict_step']) * (t['kv_bits'] / 8.0) * (1.0 - 0.5 * t['head_drop'])
    batch_penalty = 0.05 * (t['batch_size'] - 1)
    merge_bonus = 0.05 / max(1, t['merge_size'])
    latency = base + text_cost + img_cost + kv_cost + batch_penalty + merge_bonus
    return float(latency)


def fit_oracle(oracle: TorchOracle, tuples: List[Dict], lr=1e-3, steps=500, tag='oracle'):
    xs = torch.tensor([build_oracle_feature(t) for t in tuples], dtype=torch.float32)
    ys = torch.tensor([[synth_latency_fn(t)] for t in tuples], dtype=torch.float32)
    opt = torch.optim.Adam(oracle.parameters(), lr=lr)
    losses = []
    for i in range(steps):
        pred = oracle(xs)
        loss = F.mse_loss(pred, ys)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i+1) % max(1, (steps//4)) == 0:
            print(f"[Oracle] Step {i+1}/{steps} MSE={loss.item():.6f}")
        losses.append(loss.item())
    plt.figure(figsize=(4,3))
    plt.plot(losses)
    plt.xlabel('step'); plt.ylabel('MSE'); plt.title('Latency Oracle Fit')
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, f'{tag}_fit_loss.pdf'), bbox_inches='tight'); plt.close()
    return losses


class Planner(nn.Module):
    def __init__(self, oracle: TorchOracle, d_in=12):
        super().__init__()
        self.oracle = oracle
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU()
        )
        self.token_keep = nn.Linear(32, 4)
        self.merge_sz = nn.Linear(32, 3)
        self.depth_t = nn.Linear(32, 4)
        self.depth_v = nn.Linear(32, 4)
        self.precision = nn.Linear(32, 3)
        self.kv_bits = nn.Linear(32, 3)
        self.head_drop = nn.Linear(32, 3)
        self.evict = nn.Linear(32, 4)
        self.keep_vals = [0.25, 0.5, 0.75, 1.0]
        self.merge_vals = [2, 4, 8]
        self.depth_vals = [0.25, 0.5, 0.75, 1.0]
        self.prec_vals = [8, 6, 4]
        self.kv_vals = [8, 6, 4]
        self.hd_vals = [0.0, 0.25, 0.5]
        self.ev_vals = [0, 16, 32, 64]

    def _encode_features(self, stats: Dict) -> torch.Tensor:
        vec = torch.tensor([
            stats['budget_B'], stats['text_len'], stats['img_len'], stats['s_mean'], stats['s_std'],
            stats['s_q25'], stats['s_q50'], stats['s_q75'], stats['H_txt'], stats['H_img'], stats['batch_size'],
            stats.get('pattern_id', 0)
        ], dtype=torch.float32)
        return vec

    def qvi_select(self, hidden: torch.Tensor, K=6):
        logit_keep = self.token_keep(hidden)
        logit_merge = self.merge_sz(hidden)
        logit_dt = self.depth_t(hidden)
        logit_dv = self.depth_v(hidden)
        logit_prec = self.precision(hidden)
        logit_kv = self.kv_bits(hidden)
        logit_hd = self.head_drop(hidden)
        logit_ev = self.evict(hidden)
        logits_dict = {
            'keep': logit_keep.squeeze(0), 'merge': logit_merge.squeeze(0), 'dt': logit_dt.squeeze(0),
            'dv': logit_dv.squeeze(0), 'prec': logit_prec.squeeze(0), 'kv': logit_kv.squeeze(0),
            'hd': logit_hd.squeeze(0), 'ev': logit_ev.squeeze(0)
        }
        top_indices = {k: torch.topk(v, k=min(2, v.shape[-1])).indices.tolist() for k, v in logits_dict.items()}
        keys = list(top_indices.keys())
        candidates = []
        def rec_build(i, current):
            if i == len(keys):
                candidates.append(list(current))
                return
            for idx in top_indices[keys[i]]:
                current.append(idx)
                rec_build(i+1, current)
                current.pop()
        rec_build(0, [])
        scores = []
        for cand in candidates:
            s = 0.0
            for j, k in enumerate(keys):
                s += logits_dict[k][cand[j]].item()
            scores.append(s)
        order = np.argsort(scores)[::-1][:K]
        selected = [candidates[i] for i in order]
        weights = torch.ones(len(selected)) / len(selected)
        return selected, weights

    def plan_and_predict_latency(self, stats: Dict) -> Tuple[List[Tuple[Dict, float]], float]:
        with torch.no_grad():
            vec = self._encode_features(stats).unsqueeze(0)
            hidden = self.mlp(vec)
            tuples, weights = self.qvi_select(hidden, K=6)
            plans = []
            pred_lat = 0.0
            for t in range(len(tuples)):
                k_idx, m_idx, dt_idx, dv_idx, p_idx, kv_idx, hd_idx, ev_idx = tuples[t]
                feat = {
                    'text_len': stats['text_len'], 'img_len': stats['img_len'],
                    'keep_ratio': self.keep_vals[k_idx], 'merge_size': self.merge_vals[m_idx],
                    'depth_text': self.depth_vals[dt_idx], 'depth_vision': self.depth_vals[dv_idx],
                    'precision_id': self.prec_vals[p_idx], 'kv_bits': self.kv_vals[kv_idx],
                    'head_drop': self.hd_vals[hd_idx], 'evict_step': self.ev_vals[ev_idx],
                    'batch_size': stats['batch_size']
                }
                lat = self.oracle.predict(feat)
                pred_lat += weights[t].item() * lat
                plans.append((feat, weights[t].item()))
            return plans, float(pred_lat)

    def greedy_plan(self, stats: Dict, budget_B: float) -> Dict:
        vec = self._encode_features(stats).unsqueeze(0)
        hidden = self.mlp(vec)
        tuples, _ = self.qvi_select(hidden, K=12)
        best_under = None
        best_lat = float('inf')
        for t in tuples:
            k_idx, m_idx, dt_idx, dv_idx, p_idx, kv_idx, hd_idx, ev_idx = t
            feat = {
                'text_len': stats['text_len'], 'img_len': stats['img_len'],
                'keep_ratio': self.keep_vals[k_idx], 'merge_size': self.merge_vals[m_idx],
                'depth_text': self.depth_vals[dt_idx], 'depth_vision': self.depth_vals[dv_idx],
                'precision_id': self.prec_vals[p_idx], 'kv_bits': self.kv_vals[kv_idx],
                'head_drop': self.hd_vals[hd_idx], 'evict_step': self.ev_vals[ev_idx],
                'batch_size': stats['batch_size']
            }
            lat = self.oracle.predict(feat)
            if lat <= budget_B and lat < best_lat:
                best_lat = lat
                best_under = feat
        if best_under is None:
            best = None
            best_lat = float('inf')
            for t in tuples:
                k_idx, m_idx, dt_idx, dv_idx, p_idx, kv_idx, hd_idx, ev_idx = t
                feat = {
                    'text_len': stats['text_len'], 'img_len': stats['img_len'],
                    'keep_ratio': self.keep_vals[k_idx], 'merge_size': self.merge_vals[m_idx],
                    'depth_text': self.depth_vals[dt_idx], 'depth_vision': self.depth_vals[dv_idx],
                    'precision_id': self.prec_vals[p_idx], 'kv_bits': self.kv_vals[kv_idx],
                    'head_drop': self.hd_vals[hd_idx], 'evict_step': self.ev_vals[ev_idx],
                    'batch_size': stats['batch_size']
                }
                lat = self.oracle.predict(feat)
                if lat < best_lat:
                    best_lat = lat
                    best = feat
            return best
        return best_under


# -----------------------
# Utilities for stats and training
# -----------------------

def compute_sharedness_entropy(model: ToyVLM, sample: Dict) -> Dict:
    out = model.warm_pass(sample['text_len'], sample['img_len'], set(sample['decisive_indices'][0]), set(sample['decisive_indices'][1]), sample['pattern'])
    s_txt, _ = model._sharedness_js(out['attn_t2i'], out['attn_i2t'])
    H_txt = model._conditional_entropy(out['A_l_txt'], out['A_lp1_txt'])
    H_img = model._conditional_entropy(out['A_l_img'], out['A_lp1_img'])
    s_np = s_txt.detach().numpy()
    stats = {
        's_mean': float(s_np.mean()),
        's_std': float(s_np.std()),
        's_q25': float(np.quantile(s_np, 0.25)),
        's_q50': float(np.quantile(s_np, 0.5)),
        's_q75': float(np.quantile(s_np, 0.75)),
        'H_txt': float(H_txt), 'H_img': float(H_img)
    }
    return stats


def train_base_classifier(model: ToyVLM, train_samples: List[Dict], epochs=3, lr=1e-3, tag='pretrain'):
    print("[Pretrain] Training base classifier head...")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        ep_loss = 0.0
        for sample in train_samples:
            label = torch.tensor([sample['label']], dtype=torch.long)
            plan = {'keep_ratio': 1.0, 'depth_text': 1.0, 'depth_vision': 1.0, 'precision_id': 8,
                    'kv_bits': 8, 'head_drop': 0.0, 'evict_step': 0}
            logits, _ = model.forward_classify(sample, plan)
            loss = F.cross_entropy(logits.unsqueeze(0), label)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        ep_loss /= max(1, len(train_samples))
        print(f"[Pretrain] Epoch {ep+1}/{epochs} Loss={ep_loss:.4f}")
        losses.append(ep_loss)
    plt.figure(figsize=(4,3))
    plt.plot(losses, label='train_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Pretrain Loss')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f'{tag}_training_loss_qshift.pdf'), bbox_inches='tight'); plt.close()


# -----------------------
# Training entrypoint
# -----------------------

def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset_json(path: str) -> Dict[str, List[Dict]]:
    with open(path, 'r') as f:
        return json.load(f)


def main(config_path: str = None):
    if config_path is None:
        config_path = os.path.join(CONFIG_DIR, 'default.yaml')
    cfg = load_config(config_path)
    set_seed(cfg.get('seed', 42))

    dataset_path = os.path.join(DATA_DIR, cfg['data']['dataset_json'])
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset json not found at {dataset_path}. Run preprocess.py first.")
    data = load_dataset_json(dataset_path)
    train_samples = data['train']
    val_samples = data.get('val', [])

    # Build model
    model_cfg = cfg['model']
    model = ToyVLM(d_model=model_cfg['d_model'], n_heads=model_cfg['n_heads'], n_classes=cfg['data']['n_classes'])

    # Pretrain classifier
    train_base_classifier(model, train_samples, epochs=cfg['train']['pretrain_epochs'], lr=cfg['train']['pretrain_lr'], tag='pretrain')
    # Save classifier
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'toy_vlm_classifier.pt'))

    # Fit latency oracle on synthetic tuples
    print("[Oracle] Fitting latency oracle on synthetic tuples...")
    tuples = []
    for _ in range(cfg['oracle']['n_tuples']):
        feat = {
            'text_len': random.randint(cfg['data']['text_len_range'][0], cfg['data']['text_len_range'][1]),
            'img_len': random.randint(cfg['data']['img_len_range'][0], cfg['data']['img_len_range'][1]),
            'keep_ratio': random.choice([0.25, 0.5, 0.75, 1.0]),
            'merge_size': random.choice([2, 4, 8]),
            'depth_text': random.choice([0.25, 0.5, 0.75, 1.0]),
            'depth_vision': random.choice([0.25, 0.5, 0.75, 1.0]),
            'precision_id': random.choice([8, 6, 4]),
            'kv_bits': random.choice([8, 6, 4]),
            'head_drop': random.choice([0.0, 0.25, 0.5]),
            'evict_step': random.choice([0, 16, 32, 64]),
            'batch_size': 1,
        }
        tuples.append(feat)
    oracle = TorchOracle(in_dim=11, hidden=cfg['oracle']['hidden'])
    fit_oracle(oracle, tuples, lr=cfg['oracle']['lr'], steps=cfg['oracle']['steps'], tag='oracle')
    torch.save(oracle.state_dict(), os.path.join(MODELS_DIR, 'latency_oracle.pt'))

    # Planner training
    print("[Planner] Training planner (QVI-like)...")
    planner = Planner(oracle, d_in=12)
    opt = torch.optim.Adam(planner.parameters(), lr=cfg['planner']['lr'])
    budgets = cfg['planner']['budgets_ms']
    lambda_b = cfg['planner']['lambda_budget']
    plosses = []

    for ep in range(cfg['planner']['epochs']):
        ep_loss = 0.0
        for sample in train_samples:
            B = random.choice(budgets)
            sH = compute_sharedness_entropy(model, sample)
            stats = {
                'budget_B': B,
                'text_len': sample['text_len'], 'img_len': sample['img_len'], 'batch_size': 1,
                'pattern_id': 0 if sample['pattern'] == 'aligned' else (1 if sample['pattern'] == 'noisy' else 2),
                **sH
            }
            plans, pred_lat = planner.plan_and_predict_latency(stats)
            total_loss = 0.0
            for (plan, w) in plans:
                logits, _ = model.forward_classify(sample, plan)
                label = torch.tensor([sample['label']], dtype=torch.long)
                L_task = F.cross_entropy(logits.unsqueeze(0), label)
                total_loss = total_loss + w * L_task
            loss = total_loss + lambda_b * max(0.0, pred_lat - B)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item())
        ep_loss /= max(1, len(train_samples))
        print(f"[Planner] Epoch {ep+1}/{cfg['planner']['epochs']} Loss={ep_loss:.4f}")
        plosses.append(ep_loss)

    plt.figure(figsize=(4,3))
    plt.plot(plosses, label='planner_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Planner Training Loss')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'planner_training_loss_qshift.pdf'), bbox_inches='tight'); plt.close()

    # Save planner
    torch.save(planner.state_dict(), os.path.join(MODELS_DIR, 'planner.pt'))

    # Quick sanity plan on one val sample
    if len(val_samples) > 0:
        sample = val_samples[0]
    else:
        sample = train_samples[0]
    sH = compute_sharedness_entropy(model, sample)
    B = budgets[0]
    stats = {'budget_B': B, 'text_len': sample['text_len'], 'img_len': sample['img_len'], 'batch_size': 1,
             'pattern_id': 0 if sample['pattern'] == 'aligned' else (1 if sample['pattern'] == 'noisy' else 2), **sH}
    best_plan = planner.greedy_plan(stats, B)
    logits, _ = model.forward_classify(sample, best_plan)
    pred = int(logits.argmax().item())
    summary = {
        'planned_config': best_plan,
        'predicted_class': pred,
        'sharedness_mean': sH['s_mean'],
        'H_txt': sH['H_txt']
    }
    with open(os.path.join(MODELS_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("[Train] Saved models and summary.")


if __name__ == '__main__':
    cfg_path = os.path.join(CONFIG_DIR, 'default.yaml')
    main(cfg_path)
