import os
import json
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.stats import spearmanr
import pandas as pd

# -----------------------
# Paths
# -----------------------
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGES_DIR = os.path.join(PROJECT_DIR, '.research', 'iteration1', 'images')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'config')

os.makedirs(IMAGES_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Core classes (duplicated for standalone evaluation)
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
        attn = torch.einsum('htd,hkd->htk', Qh, Kh) / np.sqrt(head_dim)
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
        return {'txt': txt, 'img': img, 'A_l_txt': A_l_txt, 'A_lp1_txt': A_lp1_txt, 'A_l_img': A_l_img, 'A_lp1_img': A_lp1_img, 'attn_t2i': attn_t2i, 'attn_i2t': attn_i2t}

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
        js = 0.5 * ((p * (p.add(1e-6).log() - m.log())).sum(dim=-1) + (q * (q.add(1e-6).log() - m.log())).sum(dim=-1))
        s_txt = 1.0 - (js - js.min()) / (js.max() - js.min() + 1e-6)
        s_img = 1.0 - (js.mean())
        return s_txt, s_img

    def _apply_precision_noise(self, x, bits):
        noise = {8: 0.005, 6: 0.02, 4: 0.05}.get(bits, 0.01)
        return x + noise * torch.randn_like(x)

    def forward_classify(self, sample: Dict, plan: Dict):
        out = self.warm_pass(sample['text_len'], sample['img_len'], set(sample['decisive_indices'][0]), set(sample['decisive_indices'][1]), sample['pattern'])
        txt = out['txt']; img = out['img']; attn_t2i = out['attn_t2i']; attn_i2t = out['attn_i2t']
        s_txt, _ = self._sharedness_js(attn_t2i, attn_i2t)
        k = max(1, int(len(s_txt) * float(plan.get('keep_ratio', 1.0))))
        top_idx = torch.topk(s_txt, k=k).indices
        txt_kept = txt[top_idx]
        depth_t = plan.get('depth_text', 1.0); depth_v = plan.get('depth_vision', 1.0)
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
        return logits, out

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
        lat_per_token, kv_mem = [], []
        quality = 0.0
        backoff_triggers = 0
        kept_heads = max(1, int(self.n_heads * (1.0 - head_drop)))
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
                kv_bits = 8; precision_id = 8; keep_ratio = min(1.0, keep_ratio + 0.25)
                kept_heads = self.n_heads; backoff_triggers += 1; base_margin = 0.85
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
                candidates.append(list(current)); return
            for idx in top_indices[keys[i]]:
                current.append(idx); rec_build(i+1, current); current.pop()
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

    def plan_and_predict_latency(self, stats: Dict):
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
        best_under = None; best_lat = float('inf')
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
                best_lat = lat; best_under = feat
        if best_under is None:
            best = None; best_lat = float('inf')
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
                    best_lat = lat; best = feat
            return best
        return best_under


# -----------------------
# Evaluation utilities
# -----------------------

def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset_json(path: str) -> Dict[str, List[Dict]]:
    with open(path, 'r') as f:
        return json.load(f)


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


def eval_plan_accuracy_latency(model: ToyVLM, ds: List[Dict], plan_fn, oracle: TorchOracle, budget: float, name: str):
    correct = 0
    total = 0
    latencies = []
    for sample in ds:
        sH = compute_sharedness_entropy(model, sample)
        stats = {
            'budget_B': budget,
            'text_len': sample['text_len'], 'img_len': sample['img_len'], 'batch_size': 1,
            'pattern_id': 0 if sample['pattern'] == 'aligned' else (1 if sample['pattern'] == 'noisy' else 2),
            **sH
        }
        plan = plan_fn(stats)
        lat = oracle.predict(plan)
        latencies.append(lat)
        logits, _ = model.forward_classify(sample, plan)
        pred = logits.argmax().item()
        if pred == sample['label']:
            correct += 1
        total += 1
    accuracy = correct / max(1, total)
    avg_lat = float(np.mean(latencies))
    print(f"[{name}] Budget={budget:.3f}ms -> Acc={accuracy:.3f}, Pred_Lat(ms)={avg_lat:.3f}, N={total}")
    return accuracy, avg_lat


def run_experiment_1(model, oracle, planner, test_ds, budgets):
    print("\n=== Experiment 1: End-to-end budget-conditioned multi-axis planning ===")
    methods = {}

    def plan_full(stats):
        return {
            'text_len': stats['text_len'], 'img_len': stats['img_len'], 'batch_size': 1,
            'keep_ratio': 1.0, 'merge_size': 4, 'depth_text': 1.0, 'depth_vision': 1.0,
            'precision_id': 8, 'kv_bits': 8, 'head_drop': 0.0, 'evict_step': 0
        }

    def plan_static_ptq(stats):
        return {**plan_full(stats), 'precision_id': 8}

    def plan_depth_only(stats):
        return {**plan_full(stats), 'depth_text': 0.5, 'depth_vision': 0.5}

    def plan_static_routing(stats):
        return {**plan_full(stats), 'keep_ratio': 0.5}

    def plan_naive_combo(stats):
        return {**plan_full(stats), 'keep_ratio': 0.5, 'depth_text': 0.75, 'depth_vision': 0.75, 'precision_id': 8}

    def plan_qshift(stats):
        return planner.greedy_plan(stats, stats['budget_B'])

    results = []
    for B in budgets:
        for name, fn in [
            ("Full_FP", plan_full),
            ("Static_PTQ", plan_static_ptq),
            ("Depth_Only", plan_depth_only),
            ("Static_Routing", plan_static_routing),
            ("Naive_Combo", plan_naive_combo),
            ("QSHIFT", plan_qshift)
        ]:
            acc, lat = eval_plan_accuracy_latency(model, test_ds, lambda st, fn=fn, B=B: fn({**st, 'budget_B': B}), oracle, B, name)
            results.append({'method': name, 'budget': B, 'accuracy': acc, 'latency': lat})

    df = pd.DataFrame(results)
    plt.figure(figsize=(5,4))
    sns.lineplot(data=df, x='latency', y='accuracy', hue='method', style='method', markers=True, dashes=False)
    plt.title('Accuracy vs Latency (Toy)')
    plt.xlabel('Predicted latency (ms)'); plt.ylabel('Accuracy')
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, 'accuracy_vs_latency_qshift.pdf'), bbox_inches='tight'); plt.close()
    print("[Exp1] Saved accuracy_vs_latency_qshift.pdf")


def token_importance_loo(model: ToyVLM, sample: Dict, max_tokens_check=20):
    base_plan = {'keep_ratio': 1.0, 'depth_text': 1.0, 'depth_vision': 1.0, 'precision_id': 8,
                 'kv_bits': 8, 'head_drop': 0.0, 'evict_step': 0}
    logits_full, aux = model.forward_classify(sample, base_plan)
    label = torch.tensor([sample['label']], dtype=torch.long)
    loss_full = F.cross_entropy(logits_full.unsqueeze(0), label).item()
    s_txt, _ = model._sharedness_js(aux['attn_t2i'], aux['attn_i2t'])
    T = len(s_txt)
    idxs = torch.topk(s_txt, k=min(max_tokens_check, T)).indices.tolist()
    imp = []
    for _ in idxs:
        plan = {**base_plan, 'keep_ratio': max(0.1, 1.0 - 1.0 / T)}
        logits, _ = model.forward_classify(sample, plan)
        loss = F.cross_entropy(logits.unsqueeze(0), label).item()
        imp.append(max(0.0, loss - loss_full))
    return idxs, imp, s_txt


def run_experiment_2(model: ToyVLM, ds: List[Dict]):
    print("\n=== Experiment 2: Probing sharedness and conditional entropy signals ===")
    spearmans = []
    rocs = []
    bit_deg_lowH = []
    bit_deg_highH = []

    H_txt_all = []
    for sample in ds:
        sH_all = compute_sharedness_entropy(model, sample)
        H_txt_all.append(sH_all['H_txt'])
    H_txt_med = np.median(H_txt_all)

    for idx, sample in enumerate(ds[:60]):
        sH = compute_sharedness_entropy(model, sample)
        loo_idx, imp, s_txt_full = token_importance_loo(model, sample, max_tokens_check=10)
        s_vals = s_txt_full[loo_idx].detach().numpy()
        imp = np.array(imp) + 1e-6
        rho, _ = spearmanr(s_vals, imp)
        if not np.isnan(rho):
            spearmans.append(rho)
        k = max(1, int(len(imp) * 0.3))
        labels = np.zeros(len(imp))
        labels[np.argsort(imp)[-k:]] = 1
        try:
            auc = roc_auc_score(labels, s_vals)
            rocs.append(auc)
        except Exception:
            pass
        plan8 = {'keep_ratio': 1.0, 'depth_text': 1.0, 'depth_vision': 1.0, 'precision_id': 8, 'kv_bits': 8, 'head_drop': 0.0, 'evict_step': 0}
        plan4 = {**plan8, 'precision_id': 4}
        log8, _ = model.forward_classify(sample, plan8)
        log4, _ = model.forward_classify(sample, plan4)
        def margin(x):
            v, _ = torch.topk(x, k=2)
            return (v[0] - v[1]).item()
        deg = max(0.0, margin(log8) - margin(log4))
        if sH['H_txt'] < H_txt_med:
            bit_deg_lowH.append(deg)
        else:
            bit_deg_highH.append(deg)

    print(f"[Exp2] Spearman mean={np.mean(spearmans):.3f}, ROC-AUC mean={np.mean(rocs):.3f}")

    plt.figure(figsize=(5,3))
    sns.kdeplot(spearmans, fill=True, label='Spearman')
    plt.axvline(np.mean(spearmans), color='k', linestyle='--', label=f"mean={np.mean(spearmans):.2f}")
    plt.xlabel('Spearman correlation'); plt.title('Sharedness vs Importance')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, 'signal_correlation.pdf'), bbox_inches='tight'); plt.close()

    plt.figure(figsize=(4,3))
    data = pd.DataFrame({
        'deg': bit_deg_lowH + bit_deg_highH,
        'bin': ['low_entropy']*len(bit_deg_lowH) + ['high_entropy']*len(bit_deg_highH)
    })
    sns.boxplot(data=data, x='bin', y='deg')
    plt.title('Degradation (8->4 bits) vs Entropy Bin')
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, 'bitwidth_degradation.pdf'), bbox_inches='tight'); plt.close()
    print("[Exp2] Saved signal_correlation.pdf and bitwidth_degradation.pdf.")


def run_experiment_3(model: ToyVLM, ds: List[Dict], oracle_A100: TorchOracle, planner_A100: Planner):
    print("\n=== Experiment 3: Decoder-time adaptivity (KV), device generalization, safety backoff ===")
    # KV schedule curves on first sample
    if len(ds) == 0:
        return
    sample = ds[0]
    plan_fp = {'keep_ratio': 1.0, 'depth_text': 1.0, 'depth_vision': 1.0, 'precision_id': 8, 'kv_bits': 8, 'head_drop': 0.0, 'evict_step': 0}
    plan_int8 = {**plan_fp, 'kv_bits': 8}
    plan_q = {'keep_ratio': 0.75, 'depth_text': 0.75, 'depth_vision': 0.75, 'precision_id': 6, 'kv_bits': 6, 'head_drop': 0.25, 'evict_step': 32}
    safety = {'margin_thr': 0.45}
    curves = []
    for plan in [plan_fp, plan_int8, plan_q]:
        lat, mem, q, back = model.decode_with_kv(sample, plan, gen_len=48, safety=safety)
        curves.append((plan, lat, mem, q, back))

    # Plot latency per token
    plt.figure(figsize=(5,3))
    for (plan, lat, _, _, _) in curves:
        label = f"kv{plan['kv_bits']}_prec{plan['precision_id']}_keep{plan['keep_ratio']}_hd{plan['head_drop']}"
        plt.plot(lat, label=label)
    plt.xlabel('Step'); plt.ylabel('Latency (ms)'); plt.title('Decoding Latency/step')
    plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, 'latency_per_token_qshift.pdf'), bbox_inches='tight'); plt.close()

    # Plot KV memory per step
    plt.figure(figsize=(5,3))
    for (plan, _, mem, _, _) in curves:
        label = f"kv{plan['kv_bits']}_prec{plan['precision_id']}_keep{plan['keep_ratio']}_hd{plan['head_drop']}"
        plt.plot(mem, label=label)
    plt.xlabel('Step'); plt.ylabel('KV Mem (MB)'); plt.title('KV memory/step')
    plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, 'kv_memory_per_step_qshift.pdf'), bbox_inches='tight'); plt.close()
    print("[Exp3] Saved latency_per_token_qshift.pdf and kv_memory_per_step_qshift.pdf")

    # Device generalization: Fit a T4-like oracle (scaled synthetic)
    def synth_latency_fn(features: Dict) -> float:
        base = 0.2
        text_cost = 0.01 * features['text_len'] * (0.5 + 0.5 * features['depth_text']) * (features['precision_id'] / 8.0)
        img_cost = 0.008 * features['img_len'] * features['keep_ratio'] * (0.5 + 0.5 * features['depth_vision']) * (features['precision_id'] / 8.0)
        kv_cost = 0.003 * max(1, features['evict_step']) * (features['kv_bits'] / 8.0) * (1.0 - 0.5 * features['head_drop'])
        batch_penalty = 0.05 * (features['batch_size'] - 1)
        merge_bonus = 0.05 / max(1, features['merge_size'])
        return float(base + text_cost + img_cost + kv_cost + batch_penalty + merge_bonus)

    def synth_latency_fn_T4(features: Dict) -> float:
        return 1.5 * synth_latency_fn(features) + 0.1

    tuples = []
    for _ in range(100):
        feat = {
            'text_len': random.randint(24, 64), 'img_len': random.randint(32, 96),
            'keep_ratio': random.choice([0.25, 0.5, 0.75, 1.0]), 'merge_size': random.choice([2, 4, 8]),
            'depth_text': random.choice([0.25, 0.5, 0.75, 1.0]), 'depth_vision': random.choice([0.25, 0.5, 0.75, 1.0]),
            'precision_id': random.choice([8, 6, 4]), 'kv_bits': random.choice([8, 6, 4]),
            'head_drop': random.choice([0.0, 0.25, 0.5]), 'evict_step': random.choice([0, 16, 32, 64]),
            'batch_size': 1
        }
        tuples.append(feat)
    xs = torch.tensor([[float(feat['text_len']), float(feat['img_len']), float(feat['keep_ratio']), float(feat['merge_size']), float(feat['depth_text']), float(feat['depth_vision']), float(feat['precision_id']), float(feat['kv_bits']), float(feat['head_drop']), float(feat['evict_step']), float(feat['batch_size'])] for feat in tuples], dtype=torch.float32)
    ys_T4 = torch.tensor([[synth_latency_fn_T4(t)] for t in tuples], dtype=torch.float32)
    oracle_T4 = TorchOracle(in_dim=11, hidden=64)
    opt = torch.optim.Adam(oracle_T4.parameters(), lr=5e-3)
    for _ in range(300):
        pred = oracle_T4(xs)
        loss = F.mse_loss(pred, ys_T4)
        opt.zero_grad(); loss.backward(); opt.step()

    # Budget adherence on T4 using planner trained on A100
    test_ds = ds
    budgets = [2.5, 2.0, 1.6]
    adherence_records = []
    for B in budgets:
        errors = []
        for sample in test_ds:
            sH = compute_sharedness_entropy(model, sample)
            stats = {'budget_B': B, 'text_len': sample['text_len'], 'img_len': sample['img_len'], 'batch_size': 1,
                     'pattern_id': 0 if sample['pattern'] == 'aligned' else (1 if sample['pattern'] == 'noisy' else 2), **sH}
            plan = planner_A100.greedy_plan(stats, B)
            lat_T4 = oracle_T4.predict(plan)
            err = abs(lat_T4 - B)/B
            errors.append(err)
        adherence = np.mean([1.0 if e < 0.05 else 0.0 for e in errors])
        adherence_records.append({'device': 'T4', 'budget': B, 'adherence': adherence})
        print(f"[Exp3] Device generalization to T4: Budget={B:.2f}ms adherence={adherence*100:.1f}%")

    df = pd.DataFrame(adherence_records)
    plt.figure(figsize=(4,3))
    sns.barplot(data=df, x='budget', y='adherence')
    plt.ylim(0,1)
    plt.title('Budget Adherence on T4 (planner trained on A100)')
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, 'budget_adherence_devices.pdf'), bbox_inches='tight'); plt.close()
    print("[Exp3] Saved budget_adherence_devices.pdf")


def confusion_matrix_plot(model: ToyVLM, ds: List[Dict]):
    y_true, y_pred = [], []
    plan = {'keep_ratio': 1.0, 'depth_text': 1.0, 'depth_vision': 1.0, 'precision_id': 8, 'kv_bits': 8, 'head_drop': 0.0, 'evict_step': 0}
    for sample in ds:
        logits, _ = model.forward_classify(sample, plan)
        y_true.append(sample['label'])
        y_pred.append(int(logits.argmax().item()))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (Toy)')
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, 'confusion_matrix_baseline.pdf'), bbox_inches='tight'); plt.close()
    print("[Info] Saved confusion_matrix_baseline.pdf")


def main(config_path: str = None):
    if config_path is None:
        config_path = os.path.join(CONFIG_DIR, 'default.yaml')
    cfg = load_config(config_path)
    set_seed(cfg.get('seed', 42))

    dataset_path = os.path.join(DATA_DIR, cfg['data']['dataset_json'])
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset json not found at {dataset_path}. Run preprocess.py first.")
    data = load_dataset_json(dataset_path)
    test_ds = data.get('test', [])
    train_ds = data.get('train', [])

    # Build model and load weights
    model_cfg = cfg['model']
    model = ToyVLM(d_model=model_cfg['d_model'], n_heads=model_cfg['n_heads'], n_classes=cfg['data']['n_classes'])
    model_path = os.path.join(MODELS_DIR, 'toy_vlm_classifier.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Missing trained classifier. Run train.py.")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Load oracle and planner
    oracle = TorchOracle(in_dim=11, hidden=cfg['oracle']['hidden'])
    oracle_path = os.path.join(MODELS_DIR, 'latency_oracle.pt')
    oracle.load_state_dict(torch.load(oracle_path, map_location='cpu'))
    planner = Planner(oracle, d_in=12)
    planner_path = os.path.join(MODELS_DIR, 'planner.pt')
    planner.load_state_dict(torch.load(planner_path, map_location='cpu'))
    planner.eval()

    # Experiments
    run_experiment_1(model, oracle, planner, test_ds, budgets=cfg['planner']['budgets_ms'])
    run_experiment_2(model, train_ds[:120])
    run_experiment_3(model, test_ds[:80], oracle, planner)
    confusion_matrix_plot(model, test_ds[:100])

    print("\nAll evaluation figures saved in:", IMAGES_DIR)


if __name__ == '__main__':
    cfg_path = os.path.join(CONFIG_DIR, 'default.yaml')
    main(cfg_path)
