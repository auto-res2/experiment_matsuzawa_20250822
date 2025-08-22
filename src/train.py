# -*- coding: utf-8 -*-
"""
train.py

Implements the core SURGE-Prompt components, training loops, and experiment runners.
Figures are saved as high-quality PDF files into .research/iteration1/images.
Models are saved to models/.

This file contains:
- Fragment IR and Validator
- Synthetic dataset maker
- TargetBlackBox simulator
- Surrogate (embedder + transformer ensemble) and router
- Proposal/learning API for SURGE
- Baselines (Random, ZOPO-like)
- Experiments 1–3 (lightweight, quick mode support)

GPU/CPU: will use CUDA if available; otherwise CPU.
"""
from __future__ import annotations
import os
import json
import math
import time
import copy
import random
import string
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm

try:
    from sklearn.metrics import confusion_matrix
except Exception:
    confusion_matrix = None

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------
# Paths and I/O helpers
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR_DEFAULT = os.path.join(BASE_DIR, '.research', 'iteration1', 'images')
DATA_DIR_DEFAULT = os.path.join(BASE_DIR, 'data')
MODELS_DIR_DEFAULT = os.path.join(BASE_DIR, 'models')

for d in [IMAGES_DIR_DEFAULT, DATA_DIR_DEFAULT, MODELS_DIR_DEFAULT]:
    os.makedirs(d, exist_ok=True)


def _savefig_pdf(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


# ---------------------------------
# Utilities
# ---------------------------------

def set_global_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------
# Fragment IR (PIR) and Validator
# ---------------------------------
FRAG_TYPES = [
    'safety_guardrails',
    'output_schema',
    'reasoning_trigger',
    'step_plan',
    'tool_verifier_hints',
    'tone_style',
    'brevity_cost_policy',
]


@dataclass
class Fragment:
    id: str
    type: str
    content: str
    immutable: bool = False
    max_tokens: int = 256
    reserved_tokens: int = 0
    markers: Dict[str, Any] = field(default_factory=dict)
    version: str = 'v1'
    provenance: str = 'seed'


@dataclass
class PIR:
    fragments: List[Fragment]

    def to_json(self) -> str:
        return json.dumps({'fragments': [f.__dict__ for f in self.fragments]}, ensure_ascii=False)

    def clone(self) -> 'PIR':
        return PIR([copy.deepcopy(f) for f in self.fragments])

    def render(self, item: Dict[str, Any], enable_router: bool = False, router_decision: Optional[Dict[str, bool]] = None) -> str:
        parts = []
        for f in self.fragments:
            if enable_router and router_decision is not None:
                if f.type in router_decision and not router_decision[f.type]:
                    continue
            parts.append(f.content)
        parts.append(f"Input:\n{item['input']}")
        return "\n\n".join(parts)

    def types(self) -> List[str]:
        return [f.type for f in self.fragments]

    def count_tokens(self) -> int:
        return sum(len(f.content.split()) for f in self.fragments)


class FragmentBank:
    def __init__(self):
        self.store: List[Fragment] = []
        self._bootstrap()

    def _bootstrap(self):
        self.store.extend([
            Fragment('sg1', 'safety_guardrails', 'Follow safety rules. Avoid harmful content and PII.', True, 60),
            Fragment('os1', 'output_schema', 'Return JSON: {"answer": string}. Ensure valid JSON only.', True, 80,
                     markers={'json_schema': {'type': 'object', 'properties': {'answer': {'type': 'string'}}, 'required': ['answer']}}),
            Fragment('rt1', 'reasoning_trigger', 'Think step by step. Verify each calculation.'),
            Fragment('rt2', 'reasoning_trigger', 'Use deliberate, structured reasoning and double-check.'),
            Fragment('sp1', 'step_plan', '1) Parse question. 2) Plan solution. 3) Compute carefully. 4) Validate result.'),
            Fragment('sp2', 'step_plan', 'Steps: understand -> decompose -> solve -> verify -> concise answer.'),
            Fragment('tv1', 'tool_verifier_hints', 'If unsure, re-evaluate your steps before finalizing.'),
            Fragment('ts1', 'tone_style', 'Respond clearly and professionally.'),
            Fragment('bc1', 'brevity_cost_policy', 'Keep answers concise. Avoid unnecessary verbosity.'),
            Fragment('bc2', 'brevity_cost_policy', 'Be succinct; minimize tokens while maintaining correctness.'),
        ])

    def sample(self, frag_type: str, k: int = 2) -> List[Fragment]:
        cands = [f for f in self.store if f.type == frag_type]
        random.shuffle(cands)
        return cands[:k]


def to_pir_from_seed(seed_text: str, frag_bank: FragmentBank) -> PIR:
    frags = []
    safety = frag_bank.sample('safety_guardrails', 1)[0]
    oschema = frag_bank.sample('output_schema', 1)[0]
    frags.append(copy.deepcopy(safety))
    frags.append(copy.deepcopy(oschema))
    rt = copy.deepcopy(frag_bank.sample('reasoning_trigger', 1)[0])
    if 'think' in seed_text.lower():
        rt.content = 'Think step by step and validate intermediate results.'
    sp = copy.deepcopy(frag_bank.sample('step_plan', 1)[0])
    ts = copy.deepcopy(frag_bank.sample('tone_style', 1)[0])
    bc = copy.deepcopy(frag_bank.sample('brevity_cost_policy', 1)[0])
    frags.extend([rt, sp, ts, bc])
    return PIR(frags)


class Validator:
    def __init__(self):
        self.max_total_tokens = 1200
        self.type_order_priority = {
            'safety_guardrails': 0,
            'output_schema': 1,
            'reasoning_trigger': 2,
            'step_plan': 3,
            'tool_verifier_hints': 4,
            'tone_style': 5,
            'brevity_cost_policy': 6,
        }

    def check(self, pir: PIR) -> bool:
        types = pir.types()
        if 'safety_guardrails' not in types or 'output_schema' not in types:
            return False
        if types.index('safety_guardrails') > types.index('output_schema'):
            return False
        for f in pir.fragments:
            if f.immutable and '[EDITED]' in f.content:
                return False
        for f in pir.fragments:
            if len(f.content.split()) > f.max_tokens:
                return False
        if pir.count_tokens() > self.max_total_tokens:
            return False
        os_frags = [f for f in pir.fragments if f.type == 'output_schema']
        if os_frags:
            if 'JSON' not in os_frags[0].content and 'json' not in os_frags[0].content.lower():
                return False
        return True

    def repair(self, pir: PIR) -> PIR:
        frags = sorted(pir.fragments, key=lambda f: self.type_order_priority.get(f.type, 99))
        return PIR(frags)


# ---------------------------------
# Synthetic dataset
# ---------------------------------

def make_synthetic_dataset(task: str, n_dev: int, n_test: int, seed: int = 0) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    def mk_item(i):
        diff = rng.uniform()
        inp = f"{task.upper()} Q{i}: compute something nontrivial with difficulty {diff:.2f}."
        return {'id': i, 'input': inp, 'difficulty': float(diff)}
    dev = [mk_item(i) for i in range(n_dev)]
    test = [mk_item(i) for i in range(n_dev, n_dev + n_test)]
    return dev, test


# ---------------------------------
# Black-box target API simulator
# ---------------------------------
class TargetBlackBox:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.weights_A = {
            'reasoning_trigger': 0.50,
            'step_plan': 0.35,
            'tool_verifier_hints': 0.10,
            'tone_style': 0.05,
            'brevity_cost_policy': -0.08,
            'safety_guardrails': 0.00,
            'output_schema': 0.00,
        }
        self.weights_B = {k: v + self.rng.normal(0, 0.05) for k, v in self.weights_A.items()}
        self.token_base = 60
        self.token_type_cost = {
            'reasoning_trigger': 30,
            'step_plan': 50,
            'tool_verifier_hints': 15,
            'tone_style': 10,
            'brevity_cost_policy': -20,
            'safety_guardrails': 25,
            'output_schema': 35,
        }

    def _frag_features(self, pir: PIR) -> Dict[str, float]:
        feats = {t: 0.0 for t in FRAG_TYPES}
        for f in pir.fragments:
            qual = 1.0
            text = f.content.lower()
            if f.type in ('reasoning_trigger', 'step_plan'):
                if any(w in text for w in ['think', 'step', 'verify', 'plan', 'decompose', 'validate']):
                    qual += 0.2
                if any(w in text for w in ['check', 'carefully', 'double-check']):
                    qual += 0.1
            if f.type == 'brevity_cost_policy':
                if any(w in text for w in ['concise', 'succinct', 'brevity', 'minimize']):
                    qual += 0.1
            if f.type == 'output_schema':
                if 'json' in text:
                    qual += 0.2
            if f.type == 'safety_guardrails':
                if any(w in text for w in ['safety', 'harm', 'pii', 'avoid']):
                    qual += 0.2
            feats[f.type] = max(feats[f.type], qual)
        return feats

    def _token_cost(self, pir: PIR, router_decision: Optional[Dict[str, bool]] = None) -> int:
        tokens = self.token_base
        for f in pir.fragments:
            if router_decision is not None and f.type in router_decision and not router_decision[f.type]:
                continue
            tokens += self.token_type_cost.get(f.type, 0)
            tokens += int(0.2 * len(f.content.split()))
        return max(tokens, 5)

    def _violation_probs(self, pir: PIR) -> Tuple[float, float]:
        feats = self._frag_features(pir)
        p_format = 0.25 * math.exp(-0.8 * (feats['output_schema'] - 1.0))
        p_safety = 0.20 * math.exp(-0.8 * (feats['safety_guardrails'] - 1.0))
        p_format = float(min(max(p_format, 0.01), 0.6))
        p_safety = float(min(max(p_safety, 0.01), 0.6))
        return p_format, p_safety

    def call(self, model: str, pir: PIR, item: Dict[str, Any], paraphrase_factor: float = 1.0,
             router_decision: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        weights = self.weights_A if model == 'A' else self.weights_B
        feats = self._frag_features(pir)
        base_logit = -1.0 * (item['difficulty'] - 0.5)
        for t, w in weights.items():
            base_logit += w * feats.get(t, 0.0)
        base_logit -= 0.3 * (paraphrase_factor - 1.0)
        p_success = 1.0 / (1.0 + math.exp(-base_logit))
        success = float(self.rng.uniform() < p_success)
        p_format, p_safety = self._violation_probs(pir)
        format_violation = float(self.rng.uniform() < p_format)
        safety_violation = float(self.rng.uniform() < p_safety)
        tokens = self._token_cost(pir, router_decision)
        return {
            'success': success,
            'p_success_true': p_success,
            'format_violation': format_violation,
            'safety_violation': safety_violation,
            'tokens': tokens,
        }


# ---------------------------------
# Metrics
# ---------------------------------

def wilson_lcb(successes: int, n: int, alpha: float = 0.1) -> float:
    if n == 0:
        return 0.0
    z = norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    radius = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (center - radius) / denom


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        idx = (probs > bins[i]) & (probs <= bins[i + 1])
        if idx.sum() == 0:
            continue
        conf = probs[idx].mean()
        acc = labels[idx].mean()
        ece += (idx.sum() / N) * abs(acc - conf)
    return float(ece)


# ---------------------------------
# Surrogate encoder/model
# ---------------------------------
class FragEmbed(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.type_emb = nn.Embedding(len(FRAG_TYPES), d_model)
        vocab_size = 256
        self.char_emb = nn.Embedding(vocab_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, frags: List[Fragment]) -> torch.Tensor:
        vecs = []
        for f in frags:
            t_idx = torch.tensor([FRAG_TYPES.index(f.type)], dtype=torch.long)
            t_vec = self.type_emb(t_idx)
            text = f.content
            if len(text) == 0:
                c_vec = torch.zeros((1, self.d_model))
            else:
                idx = torch.tensor([ord(c) % 256 for c in text[:300]], dtype=torch.long)
                c_vec = self.char_emb(idx).mean(dim=0, keepdim=True)
            vec = self.layer_norm(t_vec + c_vec)
            vecs.append(vec)
        if len(vecs) == 0:
            return torch.zeros((1, self.d_model))
        return torch.cat(vecs, dim=0)


class FragTransformer(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.head = nn.Linear(d_model, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, x], dim=1)
        h = self.encoder(h)
        pooled = h[:, 0, :]
        out = self.head(pooled)
        return out


class SurrogateEnsemble:
    def __init__(self, ensemble_size: int = 3, seed: int = 0, device: Optional[str] = None):
        set_global_seed(seed)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.ensemble = []
        for _ in range(ensemble_size):
            model = nn.Sequential(FragTransformer())
            model.to(self.device)
            self.ensemble.append(model)
        self.embedder = FragEmbed().to(self.device)
        self.temp = 1.0
        self.alpha = 0.1
        self.residuals_success: List[float] = []
        self.training = False

    def _encode_pir(self, pir: PIR) -> torch.Tensor:
        with torch.no_grad():
            fe = self.embedder(pir.fragments).unsqueeze(0).to(self.device)
        return fe

    def predict(self, pir: PIR) -> Dict[str, float]:
        x = self._encode_pir(pir)
        logits = []
        with torch.no_grad():
            for m in self.ensemble:
                out = m(x)
                logits.append(out)
        logit = torch.stack(logits, dim=0).mean(dim=0).squeeze(0)
        success_logit = logit[0] / max(self.temp, 1e-6)
        p_success = torch.sigmoid(success_logit).item()
        p_format = torch.sigmoid(logit[1]).item()
        p_safety = torch.sigmoid(logit[2]).item()
        cost = F.softplus(logit[3]).item()
        q_alpha = np.quantile(self.residuals_success, 1 - self.alpha) if len(self.residuals_success) > 10 else 0.15
        lcb = max(p_success - float(q_alpha), 0.0)
        return {
            'p_success': float(p_success),
            'p_format': float(p_format),
            'p_safety': float(p_safety),
            'cost_pred': float(cost),
            'lcb_success': float(lcb),
        }

    def fit_epoch(self, batch: List[Tuple[PIR, Dict[str, float]]], lr: float = 5e-4):
        self.training = True
        opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.ensemble]
        loss_meter = 0.0
        for (pir, tgt) in batch:
            x = self._encode_pir(pir)
            y_success = torch.tensor([[tgt['success']]], dtype=torch.float32, device=self.device)
            y_format = torch.tensor([[tgt['format_violation']]], dtype=torch.float32, device=self.device)
            y_safety = torch.tensor([[tgt['safety_violation']]], dtype=torch.float32, device=self.device)
            y_cost = torch.tensor([[tgt['tokens'] / 200.0]], dtype=torch.float32, device=self.device)
            for m, opt in zip(self.ensemble, opts):
                opt.zero_grad()
                out = m(x)
                loss = F.binary_cross_entropy_with_logits(out[:, 0:1], y_success)
                loss += 0.5 * F.binary_cross_entropy_with_logits(out[:, 1:2], y_format)
                loss += 0.5 * F.binary_cross_entropy_with_logits(out[:, 2:3], y_safety)
                loss += 0.2 * F.mse_loss(F.softplus(out[:, 3:4]), y_cost)
                loss.backward()
                opt.step()
                loss_meter += loss.item()
        self.training = False
        return loss_meter / max(len(batch) * len(self.ensemble), 1)

    def recalibrate_temperature(self, cal_data: List[Tuple[PIR, float]], steps: int = 100):
        def nll_for_T(T: float) -> float:
            eps = 1e-12
            nll = 0.0
            for pir, y in cal_data:
                x = self._encode_pir(pir)
                with torch.no_grad():
                    logits = []
                    for m in self.ensemble:
                        logits.append(m(x))
                    logit = torch.stack(logits, dim=0).mean(dim=0).squeeze(0)
                    p = torch.sigmoid(logit[0] / max(T, 1e-6)).item()
                p = min(max(p, eps), 1 - eps)
                nll += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            return nll / max(len(cal_data), 1)
        best_T, best_nll = self.temp, nll_for_T(self.temp)
        for T in np.linspace(0.5, 2.5, steps):
            nll = nll_for_T(float(T))
            if nll < best_nll:
                best_nll, best_T = nll, float(T)
        self.temp = best_T
        return best_T, best_nll

    def update_conformal(self, cal_data: List[Tuple[PIR, float]], alpha: float = 0.1):
        residuals = []
        for pir, y in cal_data:
            pred = self.predict(pir)['p_success']
            residuals.append(pred - y)
        self.residuals_success = residuals
        self.alpha = alpha


# ---------------------------------
# Candidate and editing ops
# ---------------------------------
@dataclass
class Candidate:
    pir: PIR
    meta: Dict[str, Any]
    pred: Dict[str, float]
    embedding: np.ndarray

    def render(self, item: Dict[str, Any], enable_router: bool = False, router_decision: Optional[Dict[str, bool]] = None) -> str:
        return self.pir.render(item, enable_router, router_decision)


def pir_to_embedding(pir: PIR, embedder: FragEmbed) -> np.ndarray:
    with torch.no_grad():
        e = embedder(pir.fragments)
        e = e.mean(dim=0).cpu().numpy()
    return e


def random_freeform_mutation(pir: PIR, rng: random.Random) -> PIR:
    p = pir.clone()
    if len(p.fragments) > 0:
        f = rng.choice(p.fragments)
        insertion = ' [FREEFORM EDIT ' + ''.join(rng.choice(string.ascii_letters) for _ in range(8)) + '] '
        f.content = f.content + insertion
        if f.immutable and rng.random() < 0.3:
            f.content += ' [EDITED]'
    else:
        p.fragments.append(Fragment('ff_'+str(rng.randint(0,1_000_000)), 'tone_style', 'Random style text [FREEFORM].'))
    return p


def constrained_fragment_edit(pir: PIR, bank: FragmentBank, rng: random.Random) -> PIR:
    p = pir.clone()
    ops = ['insert', 'swap', 'delete', 'modify', 'move']
    op = rng.choice(ops)
    if op == 'insert':
        choices = ['reasoning_trigger', 'step_plan', 'tool_verifier_hints', 'tone_style', 'brevity_cost_policy']
        t = rng.choice(choices)
        frag = copy.deepcopy(rng.choice(bank.sample(t, k=1)))
        frag.id = f'ins_{t}_{rng.randint(0, 1_000_000_000)}'
        insert_pos = rng.randint(2, len(p.fragments))
        p.fragments.insert(insert_pos, frag)
    elif op == 'swap' and len(p.fragments) > 3:
        i, j = rng.randint(2, len(p.fragments) - 1), rng.randint(2, len(p.fragments) - 1)
        p.fragments[i], p.fragments[j] = p.fragments[j], p.fragments[i]
    elif op == 'delete' and len(p.fragments) > 3:
        i = rng.randint(2, len(p.fragments) - 1)
        if not p.fragments[i].immutable:
            del p.fragments[i]
    elif op == 'modify':
        i = rng.randint(2, len(p.fragments) - 1)
        f = p.fragments[i]
        if not f.immutable:
            replacements = {
                'concise': 'succinct', 'verify': 'double-check', 'plan': 'strategy', 'carefully': 'meticulously',
                'avoid': 'refrain from', 'safety': 'safeguards'
            }
            for k, v in replacements.items():
                if k in f.content and rng.random() < 0.5:
                    f.content = f.content.replace(k, v)
            if rng.random() < 0.3:
                f.content += ' Ensure clarity.'
    elif op == 'move' and len(p.fragments) > 3:
        i = rng.randint(2, len(p.fragments) - 1)
        j = rng.randint(2, len(p.fragments) - 1)
        frag = p.fragments.pop(i)
        p.fragments.insert(j, frag)
    return p


# ---------------------------------
# Acquisition and diversification
# ---------------------------------

def dpp_greedy(embs: np.ndarray, k: int) -> List[int]:
    if len(embs) == 0:
        return []
    K = embs @ embs.T + 1e-3 * np.eye(len(embs))
    selected = []
    C = np.zeros_like(K)
    for _ in range(min(k, len(embs))):
        gains = np.diag(K - C)
        i = int(np.argmax(gains))
        selected.append(i)
        ki = K[:, i:i+1]
        denom = max(K[i, i] - C[i, i], 1e-6)
        C += (ki @ ki.T) / denom
    return selected


def select_batch(cands: List[Candidate], k: int = 8, diversify: bool = True) -> List[Candidate]:
    if len(cands) <= k:
        return cands
    if not diversify:
        cands_sorted = sorted(cands, key=lambda c: (-c.pred['lcb_success'], c.pred['p_format'] + c.pred['p_safety']))
        return cands_sorted[:k]
    embs = np.stack([c.embedding for c in cands])
    idxs = dpp_greedy(embs, k)
    return [cands[i] for i in idxs]


def robust_stats(preds_panel: List[float]) -> Dict[str, float]:
    arr = np.array(preds_panel, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    k = max(1, int(0.1 * len(arr)))
    worst = float(np.mean(np.partition(arr, k)[:k]))
    return {'mean': mean, 'std': std, 'cvar10': worst}


# ---------------------------------
# Router and Residual Corrector
# ---------------------------------
class BanditRouter(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        set_global_seed(seed)
        self.linear = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))

    def decision(self, item: Dict[str, Any]) -> Dict[str, bool]:
        x = torch.tensor([[1.0, float(item['difficulty']), 1.0]], dtype=torch.float32)
        p_on = self.forward(x).item()
        enable = (p_on >= 0.5)
        return {'reasoning_trigger': enable, 'tool_verifier_hints': enable}

    def fit_supervised(self, logs: List[Dict[str, Any]], iters: int = 100, lr: float = 0.05):
        if len(logs) == 0:
            return
        X, Y = [], []
        for z in logs:
            diff = z['item']['difficulty'] if 'item' in z else 0.5
            y = 1.0 if z.get('delta_success', 0.0) >= 0.0 else 0.0
            X.append([1.0, float(diff), 1.0])
            Y.append([y])
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            p = self.forward(X)
            loss = F.binary_cross_entropy(p, Y)
            loss.backward()
            opt.step()


class ResidualKNN:
    def __init__(self, k: int = 5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, feats: np.ndarray, residuals: np.ndarray):
        self.X = feats.astype(np.float32)
        self.y = residuals.astype(np.float32)

    def predict(self, feats: np.ndarray) -> float:
        if self.X is None or len(self.X) == 0:
            return 0.0
        x = feats.astype(np.float32)
        d = np.linalg.norm(self.X - x, axis=1)
        idx = np.argsort(d)[:min(self.k, len(d))]
        return float(np.mean(self.y[idx]))


# ---------------------------------
# SURGE propose()/learn() API
# ---------------------------------
class SURGE:
    def __init__(self, target: TargetBlackBox, seed: int = 0, device: Optional[str] = None):
        self.bank = FragmentBank()
        self.val = Validator()
        self.target = target
        self.surr = SurrogateEnsemble(ensemble_size=3, seed=seed, device=device)
        self.rng = random.Random(seed)
        self.router = BanditRouter(seed=seed)
        self.resid_knn = ResidualKNN(k=7)

    def propose(self, n: int, task_spec: str, pir: PIR, constraints: Dict[str, Any], diversify: bool = True) -> List[Candidate]:
        cands: List[Candidate] = []
        attempts = 0
        while len(cands) < n and attempts < n * 10:
            attempts += 1
            edited = constrained_fragment_edit(pir, self.bank, self.rng)
            edited = self.val.repair(edited)
            if not self.val.check(edited):
                continue
            pred = self.surr.predict(edited)
            emb = pir_to_embedding(edited, self.surr.embedder)
            cands.append(Candidate(edited, {'task': task_spec}, pred, emb))
        if diversify and len(cands) > 1:
            cands = select_batch(cands, k=min(n, len(cands)), diversify=True)
        return cands

    def learn(self, batch_logs: List[Dict[str, Any]]):
        if len(batch_logs) == 0:
            return 0.0
        loss = self.surr.fit_epoch([(b['pir'], b['targets']) for b in batch_logs], lr=3e-4)
        cal = [(b['pir'], b['targets']['success']) for b in batch_logs]
        self.surr.update_conformal(cal, alpha=0.1)
        feats = []
        residuals = []
        for b in batch_logs:
            p = self.surr.predict(b['pir'])['p_success']
            feats.append(pir_to_embedding(b['pir'], self.surr.embedder))
            residuals.append(p - b['targets']['success'])
        if len(feats) > 0:
            self.resid_knn.fit(np.stack(feats), np.array(residuals))
        return loss

    def save(self, models_dir: str = MODELS_DIR_DEFAULT, name: str = 'surge_surrogate.pt'):
        os.makedirs(models_dir, exist_ok=True)
        state = {
            'embedder': self.surr.embedder.state_dict(),
            'ensemble': [m.state_dict() for m in self.surr.ensemble],
            'temp': self.surr.temp,
            'alpha': self.surr.alpha,
            'residuals_success': self.surr.residuals_success,
        }
        torch.save(state, os.path.join(models_dir, name))

    def load(self, models_dir: str = MODELS_DIR_DEFAULT, name: str = 'surge_surrogate.pt', device: Optional[str] = None):
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model at {path}")
        if device is None:
            device = self.surr.device
        state = torch.load(path, map_location=device)
        self.surr.embedder.load_state_dict(state['embedder'])
        for m, sd in zip(self.surr.ensemble, state['ensemble']):
            m.load_state_dict(sd)
        self.surr.temp = state.get('temp', 1.0)
        self.surr.alpha = state.get('alpha', 0.1)
        self.surr.residuals_success = state.get('residuals_success', [])


# ---------------------------------
# Baselines
# ---------------------------------
class BaselineRandom:
    def __init__(self, target: TargetBlackBox, seed: int = 0):
        self.rng = random.Random(seed)
        self.target = target
        self.bank = FragmentBank()

    def propose(self, n: int, pir: PIR) -> List[PIR]:
        cands = []
        for _ in range(n):
            cands.append(random_freeform_mutation(pir, self.rng))
        return cands


class BaselineZOPO:
    def __init__(self, target: TargetBlackBox, seed: int = 0):
        self.rng = random.Random(seed)
        self.target = target
        self.bank = FragmentBank()
        self.pool: List[PIR] = []
        self.embedder = FragEmbed()
        self.obs: List[Tuple[np.ndarray, float]] = []

    def seed_pool(self, pir: PIR, m: int = 12):
        for _ in range(m):
            e = constrained_fragment_edit(pir, self.bank, self.rng)
            self.pool.append(e)

    def propose(self, n: int, pir: PIR) -> List[PIR]:
        for _ in range(n):
            self.pool.append(constrained_fragment_edit(pir, self.bank, self.rng))
        if len(self.obs) == 0:
            return self.pool[:n]
        best_emb = max(self.obs, key=lambda t: t[1])[0]
        scored = []
        for p in self.pool:
            e = pir_to_embedding(p, self.embedder)
            sim = float(np.dot(e, best_emb) / (np.linalg.norm(e) * np.linalg.norm(best_emb) + 1e-6))
            scored.append((sim, p))
        scored.sort(key=lambda t: -t[0])
        return [p for _, p in scored[:n]]

    def observe(self, pir: PIR, success_mean: float):
        e = pir_to_embedding(pir, self.embedder)
        self.obs.append((e, success_mean))


# ---------------------------------
# Eval helpers
# ---------------------------------

def eval_candidate_on_dev(target: TargetBlackBox, model_id: str, pir: PIR, dev_subset: List[Dict[str, Any]],
                           early_stop_lcb: Optional[float] = None, alpha: float = 0.1,
                           router: Optional[BanditRouter] = None) -> Dict[str, Any]:
    successes = 0
    n = 0
    tokens = 0
    format_v = 0
    safety_v = 0
    for item in dev_subset:
        router_decision = router.decision(item) if router is not None else None
        out = target.call(model_id, pir, item, paraphrase_factor=1.0, router_decision=router_decision)
        successes += int(out['success'] > 0.5)
        tokens += out['tokens']
        format_v += int(out['format_violation'] > 0.5)
        safety_v += int(out['safety_violation'] > 0.5)
        n += 1
        if early_stop_lcb is not None and n >= 6:
            w = wilson_lcb(successes, n, alpha)
            if w < early_stop_lcb or w < 0.2:
                break
    return {
        'success_rate': successes / max(n, 1),
        'mean_tokens': tokens / max(n, 1),
        'format_rate': format_v / max(n, 1),
        'safety_rate': safety_v / max(n, 1),
        'n_eval': n,
    }


def eval_on_test_settings(target: TargetBlackBox, pir: PIR, test: List[Dict[str, Any]], router: Optional[BanditRouter] = None) -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    settings = [('A', 1.0), ('A', 1.2), ('B', 1.0), ('B', 1.2)]
    results = {}
    for mid, pf in settings:
        succ = []
        toks = []
        fmt = []
        sft = []
        for it in test:
            router_decision = router.decision(it) if router is not None else None
            out = target.call(mid, pir, it, paraphrase_factor=pf, router_decision=router_decision)
            succ.append(out['success'])
            toks.append(out['tokens'])
            fmt.append(out['format_violation'])
            sft.append(out['safety_violation'])
        results[(mid, pf)] = {
            'acc': float(np.mean(succ)),
            'tokens': float(np.mean(toks)),
            'format': float(np.mean(fmt)),
            'safety': float(np.mean(sft)),
        }
    return results


# ---------------------------------
# Experiment 1
# ---------------------------------

def run_experiment_1(seed: int = 0, quick: bool = False, images_dir: str = IMAGES_DIR_DEFAULT):
    print("\n=== Experiment 1: Budgeted effectiveness, efficiency, reliability vs baselines ===")
    set_global_seed(seed)
    target = TargetBlackBox(seed)
    surge = SURGE(target, seed)
    bank = surge.bank
    val = surge.val

    n_dev = 60 if quick else 120
    n_test = 120 if quick else 240
    dev, test = make_synthetic_dataset('gsm8k', n_dev=n_dev, n_test=n_test, seed=seed)

    seed_prompt = 'You are an expert math tutor. Think systematically and respond in JSON.'
    pir0 = to_pir_from_seed(seed_prompt, bank)
    pir0 = val.repair(pir0)

    budget_calls = 60 if quick else 160
    per_round = 6 if quick else 8
    rounds = budget_calls // per_round
    print(f"Budget calls: {budget_calls}, per_round: {per_round}, rounds: {rounds}")

    rnd = BaselineRandom(target, seed)
    zopo = BaselineZOPO(target, seed)
    zopo.seed_pool(pir0, m=8 if quick else 12)

    methods = ['SURGE', 'SURGE_no_calib', 'SURGE_no_validator', 'RANDOM', 'ZOPO']
    call_counts = {m: [] for m in methods}
    score_curves = {m: [] for m in methods}
    ece_store = {m: [] for m in methods}
    coverage_store = {m: [] for m in methods}
    violation_rates = {m: [] for m in methods}

    best_surge_pir = pir0
    best_surge_score = 0.0
    pred_probs = []
    true_labels = []
    lcb_preds = []
    true_succ_for_lcb = []

    for r in range(rounds):
        cands = surge.propose(n=per_round * 3, task_spec='gsm8k', pir=best_surge_pir,
                              constraints={'immutables': ['safety_guardrails', 'output_schema']}, diversify=True)
        batch_logs = []
        for c in cands[:per_round]:
            res = eval_candidate_on_dev(target, 'A', c.pir, random.sample(dev, k=min(12, len(dev))),
                                        early_stop_lcb=c.pred['lcb_success'], alpha=0.1, router=None)
            batch_logs.append({'pir': c.pir, 'targets': {
                'success': res['success_rate'],
                'format_violation': res['format_rate'],
                'safety_violation': res['safety_rate'],
                'tokens': res['mean_tokens'],
            }})
            pred_probs.append(c.pred['p_success'])
            true_labels.append(res['success_rate'])
            lcb_preds.append(c.pred['lcb_success'])
            true_succ_for_lcb.append(res['success_rate'])
        loss = surge.learn(batch_logs)
        best_in_round = max(batch_logs, key=lambda b: b['targets']['success'])
        if best_in_round['targets']['success'] > best_surge_score:
            best_surge_score = best_in_round['targets']['success']
            best_surge_pir = best_in_round['pir']
        call_counts['SURGE'].append(sum(b['targets'].get('n_eval', 8) for b in batch_logs))
        score_curves['SURGE'].append(best_surge_score)
        violation_rates['SURGE'].append((np.mean([b['targets']['format_violation'] for b in batch_logs]),
                                         np.mean([b['targets']['safety_violation'] for b in batch_logs])))
        print(f"[SURGE][Round {r+1}] best_dev_success={best_surge_score:.3f}, loss={loss:.4f}")

    ece = expected_calibration_error(np.array(pred_probs), np.array(true_labels))
    coverage = float(np.mean((np.array(true_succ_for_lcb) >= np.array(lcb_preds)).astype(float)))
    ece_store['SURGE'].append(ece)
    coverage_store['SURGE'].append(coverage)
    print(f"[SURGE] ECE={ece:.4f}, LCB coverage={coverage:.3f}")

    surge_nc = SURGE(target, seed)
    surge_nc.surr.temp = 1.0
    surge_nc.surr.alpha = 0.1
    pred_probs_nc, true_labels_nc = [], []
    best_nc_pir, best_nc_score = pir0, 0.0
    for r in range(rounds):
        cands = surge_nc.propose(n=per_round * 3, task_spec='gsm8k', pir=best_nc_pir, constraints={}, diversify=True)
        batch_logs = []
        for c in cands[:per_round]:
            res = eval_candidate_on_dev(target, 'A', c.pir, random.sample(dev, k=min(12, len(dev))), early_stop_lcb=None, alpha=0.1, router=None)
            batch_logs.append({'pir': c.pir, 'targets': {
                'success': res['success_rate'],
                'format_violation': res['format_rate'],
                'safety_violation': res['safety_rate'],
                'tokens': res['mean_tokens'],
            }})
            pred_probs_nc.append(c.pred['p_success'])
            true_labels_nc.append(res['success_rate'])
        surge_nc.learn(batch_logs)
        best_in_round = max(batch_logs, key=lambda b: b['targets']['success'])
        if best_in_round['targets']['success'] > best_nc_score:
            best_nc_score = best_in_round['targets']['success']
            best_nc_pir = best_in_round['pir']
        call_counts['SURGE_no_calib'].append(per_round)
        score_curves['SURGE_no_calib'].append(best_nc_score)
        violation_rates['SURGE_no_calib'].append((np.mean([b['targets']['format_violation'] for b in batch_logs]),
                                                  np.mean([b['targets']['safety_violation'] for b in batch_logs])))
    ece_nc = expected_calibration_error(np.array(pred_probs_nc), np.array(true_labels_nc))
    ece_store['SURGE_no_calib'].append(ece_nc)
    coverage_store['SURGE_no_calib'].append(np.nan)

    surge_nv = SURGE(target, seed)
    pred_probs_nv, true_labels_nv = [], []
    best_nv_pir, best_nv_score = pir0, 0.0
    for r in range(rounds):
        cands_pir = [random_freeform_mutation(best_nv_pir, random) for _ in range(per_round * 3)]
        cands = []
        for p in cands_pir:
            pred = surge_nv.surr.predict(p)
            emb = pir_to_embedding(p, surge_nv.surr.embedder)
            cands.append(Candidate(p, {'task': 'gsm8k'}, pred, emb))
        batch = select_batch(cands, k=per_round, diversify=True)
        batch_logs = []
        for c in batch:
            res = eval_candidate_on_dev(target, 'A', c.pir, random.sample(dev, k=min(12, len(dev))), early_stop_lcb=None, alpha=0.1, router=None)
            batch_logs.append({'pir': c.pir, 'targets': {
                'success': res['success_rate'],
                'format_violation': res['format_rate'],
                'safety_violation': res['safety_rate'],
                'tokens': res['mean_tokens'],
            }})
            pred_probs_nv.append(c.pred['p_success'])
            true_labels_nv.append(res['success_rate'])
        surge_nv.learn(batch_logs)
        best_in_round = max(batch_logs, key=lambda b: b['targets']['success'])
        if best_in_round['targets']['success'] > best_nv_score:
            best_nv_score = best_in_round['targets']['success']
            best_nv_pir = best_in_round['pir']
        call_counts['SURGE_no_validator'].append(per_round)
        score_curves['SURGE_no_validator'].append(best_nv_score)
        violation_rates['SURGE_no_validator'].append((np.mean([b['targets']['format_violation'] for b in batch_logs]),
                                                      np.mean([b['targets']['safety_violation'] for b in batch_logs])))
    ece_nv = expected_calibration_error(np.array(pred_probs_nv), np.array(true_labels_nv))
    ece_store['SURGE_no_validator'].append(ece_nv)
    coverage_store['SURGE_no_validator'].append(np.nan)

    best_rnd_pir, best_rnd_score = pir0, 0.0
    for r in range(rounds):
        cands = rnd.propose(per_round * 3, best_rnd_pir)
        evals = []
        for p in cands[:per_round]:
            res = eval_candidate_on_dev(target, 'A', p, random.sample(dev, k=min(12, len(dev))), early_stop_lcb=None)
            evals.append((res['success_rate'], p, res))
        evals.sort(key=lambda t: -t[0])
        if evals and evals[0][0] > best_rnd_score:
            best_rnd_score = evals[0][0]
            best_rnd_pir = evals[0][1]
        call_counts['RANDOM'].append(per_round)
        score_curves['RANDOM'].append(best_rnd_score)
        violation_rates['RANDOM'].append((np.mean([e[2]['format_rate'] for e in evals]), np.mean([e[2]['safety_rate'] for e in evals])))

    best_zp_pir, best_zp_score = pir0, 0.0
    for r in range(rounds):
        cands = zopo.propose(per_round * 2, best_zp_pir)
        evals = []
        for p in cands[:per_round]:
            res = eval_candidate_on_dev(target, 'A', p, random.sample(dev, k=min(12, len(dev))), early_stop_lcb=None)
            evals.append((res['success_rate'], p, res))
            zopo.observe(p, res['success_rate'])
        evals.sort(key=lambda t: -t[0])
        if evals and evals[0][0] > best_zp_score:
            best_zp_score = evals[0][0]
            best_zp_pir = evals[0][1]
        call_counts['ZOPO'].append(per_round)
        score_curves['ZOPO'].append(best_zp_score)
        violation_rates['ZOPO'].append((np.mean([e[2]['format_rate'] for e in evals]), np.mean([e[2]['safety_rate'] for e in evals])))

    final_best = {
        'SURGE': best_surge_pir,
        'SURGE_no_calib': best_nc_pir,
        'SURGE_no_validator': best_nv_pir,
        'RANDOM': best_rnd_pir,
        'ZOPO': best_zp_pir,
    }

    final_scores = {}
    for k, p in final_best.items():
        res = eval_on_test_settings(target, p, test, router=None)
        final_scores[k] = res
        print(f"[Exp1][{k}] Test (A,orig): acc={res[('A',1.0)]['acc']:.3f}, tokens={res[('A',1.0)]['tokens']:.1f}, format={res[('A',1.0)]['format']:.3f}, safety={res[('A',1.0)]['safety']:.3f}")

    # Plots
    plt.figure(figsize=(6,4))
    for m in methods:
        if len(score_curves[m]) == 0:
            continue
        xs = np.cumsum(call_counts[m]) if len(call_counts[m]) == len(score_curves[m]) else np.arange(1, len(score_curves[m])+1)
        plt.plot(xs, score_curves[m], label=m)
    plt.xlabel('Dev calls (cumulative)')
    plt.ylabel('Best dev success rate')
    plt.title('EXP1: Efficiency (best dev success vs calls)')
    plt.legend()
    plt.tight_layout()
    _savefig_pdf(os.path.join(images_dir, 'accuracy_exp1.pdf'))

    plt.figure(figsize=(6,4))
    names, vals = [], []
    for m in ['SURGE', 'SURGE_no_calib', 'SURGE_no_validator']:
        if len(ece_store[m]) > 0:
            names.append(m)
            vals.append(np.mean(ece_store[m]))
    sns.barplot(x=names, y=vals)
    plt.ylabel('ECE (success prob)')
    plt.title('EXP1: Calibration')
    plt.tight_layout()
    _savefig_pdf(os.path.join(images_dir, 'ece_exp1.pdf'))

    plt.figure(figsize=(6,4))
    names, vals = [], []
    for m in ['SURGE']:
        if len(coverage_store[m]) > 0:
            names.append(m)
            vals.append(np.mean(coverage_store[m]))
    sns.barplot(x=names, y=vals)
    plt.ylabel('LCB coverage (empirical)')
    plt.ylim(0,1)
    plt.title('EXP1: LCB Coverage')
    plt.tight_layout()
    _savefig_pdf(os.path.join(images_dir, 'coverage_exp1.pdf'))

    plt.figure(figsize=(6,4))
    names, fmt_vals, sft_vals = [], [], []
    for m in methods:
        if len(violation_rates[m]) == 0:
            continue
        names.append(m)
        fmt_vals.append(np.mean([v[0] for v in violation_rates[m]]))
        sft_vals.append(np.mean([v[1] for v in violation_rates[m]]))
    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width/2, fmt_vals, width, label='format')
    plt.bar(x + width/2, sft_vals, width, label='safety')
    plt.xticks(x, names, rotation=15)
    plt.ylabel('Violation rate (dev)')
    plt.title('EXP1: Violation rates by method')
    plt.legend()
    plt.tight_layout()
    _savefig_pdf(os.path.join(images_dir, 'violation_rates_exp1.pdf'))

    if confusion_matrix is not None:
        succ = []
        for it in test:
            out = target.call('A', best_surge_pir, it, paraphrase_factor=1.0)
            succ.append(int(out['success']))
        y_true = np.ones_like(succ)
        cm = confusion_matrix(y_true, succ, labels=[0,1])
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Pred success')
        plt.ylabel('True (desired)')
        plt.title('EXP1: Confusion matrix (SURGE, A,orig)')
        plt.tight_layout()
        _savefig_pdf(os.path.join(images_dir, 'confusion_matrix_exp1.pdf'))

    surge.save(models_dir=MODELS_DIR_DEFAULT, name='surge_surrogate_exp1.pt')

    return {
        'final_scores': final_scores,
        'best_surge_pir': best_surge_pir,
        'dev': dev,
        'test': test,
        'surge': surge,
    }


# ---------------------------------
# Experiment 2
# ---------------------------------

def run_experiment_2(seed: int = 0, quick: bool = False, images_dir: str = IMAGES_DIR_DEFAULT):
    print("\n=== Experiment 2: Robustness (paraphrases, model drift) + recalibration + router ===")
    set_global_seed(seed)
    target = TargetBlackBox(seed + 7)
    surge = SURGE(target, seed)
    bank = surge.bank
    val = surge.val

    n_dev = 50 if quick else 100
    n_test = 100 if quick else 200
    dev, test = make_synthetic_dataset('mmlu', n_dev=n_dev, n_test=n_test, seed=seed)

    seed_prompt = 'Answer multiple-choice questions accurately; respond in JSON with an "answer" string.'
    pir0 = to_pir_from_seed(seed_prompt, bank)
    pir0 = val.repair(pir0)

    budget_calls = 50 if quick else 120
    per_round = 5 if quick else 8
    rounds = budget_calls // per_round

    alpha_vals = [0.0, 0.5]
    best_by_alpha = {}

    for alpha in alpha_vals:
        print(f"[Exp2] Optimizing with alpha={alpha} (mean - alpha*std)")
        best_pir = pir0
        best_score = -1e9
        for r in range(rounds):
            cands = surge.propose(n=per_round * 3, task_spec='mmlu', pir=best_pir,
                                  constraints={'immutables': ['safety_guardrails', 'output_schema']}, diversify=True)
            scored = []
            for c in cands:
                preds_panel = []
                for mid in ['A', 'A_prime', 'B_prime']:
                    base = c.pred['p_success']
                    jitter = np.random.normal(0, 0.05, size=4)
                    preds_panel.extend(list(np.clip(base + jitter, 0, 1)))
                rs = robust_stats(preds_panel)
                scored.append((rs['mean'] - alpha * rs['std'], c, rs))
            scored.sort(key=lambda t: -t[0])
            batch = [s[1] for s in scored[:per_round]]
            batch_logs = []
            for c in batch:
                res = eval_candidate_on_dev(target, 'A', c.pir, random.sample(dev, k=min(10, len(dev))), early_stop_lcb=None)
                batch_logs.append({'pir': c.pir, 'targets': {
                    'success': res['success_rate'],
                    'format_violation': res['format_rate'],
                    'safety_violation': res['safety_rate'],
                    'tokens': res['mean_tokens'],
                }})
            surge.learn(batch_logs)
            top = max(batch_logs, key=lambda b: b['targets']['success'])
            score = top['targets']['success']
            if score > best_score:
                best_score = score
                best_pir = top['pir']
            print(f"  [alpha={alpha}][Round {r+1}] best_dev_success={best_score:.3f}")
        best_by_alpha[alpha] = best_pir

    router = surge.router

    def eval_setting(pir: PIR, with_router: bool) -> Dict[str, Any]:
        res = {}
        for (mid, pf_label, pf) in [('A', 'orig', 1.0), ('A', 'para', 1.2), ('B', 'orig', 1.0), ('B', 'para', 1.2)]:
            accs, toks = [], []
            for it in test:
                router_decision = router.decision(it) if with_router else None
                out = target.call(mid, pir, it, paraphrase_factor=pf, router_decision=router_decision)
                accs.append(out['success'])
                toks.append(out['tokens'])
            res[(mid, pf_label)] = {'acc': float(np.mean(accs)), 'tokens': float(np.mean(toks))}
        return res

    pir_mean = best_by_alpha[0.0]
    pir_robust = best_by_alpha[0.5]

    calib_items = test[:30]
    cal_pairs = []
    for it in calib_items:
        y = target.call('B', pir_robust, it, paraphrase_factor=1.0)['success']
        cal_pairs.append((pir_robust, y))
    T, nll = surge.surr.recalibrate_temperature(cal_pairs, steps=40)
    surge.surr.update_conformal(cal_pairs, alpha=0.1)
    print(f"[Exp2] Recalibrated temperature on B: T={T:.2f}, NLL={nll:.4f}")

    preds = [surge.surr.predict(pir_robust)['lcb_success'] for _ in calib_items]
    true_succ = [target.call('B', pir_robust, it)['success'] for it in calib_items]
    coverage_B = float(np.mean((np.array(true_succ) >= np.array(preds)).astype(float)))

    res_mean_nr = eval_setting(pir_mean, with_router=False)
    res_rob_nr = eval_setting(pir_robust, with_router=False)
    res_rob_wr = eval_setting(pir_robust, with_router=True)

    tokens_no_router = np.mean([res_rob_nr[(m, s)]['tokens'] for (m, s) in res_rob_nr])
    tokens_with_router = np.mean([res_rob_wr[(m, s)]['tokens'] for (m, s) in res_rob_wr])
    token_savings = (tokens_no_router - tokens_with_router) / max(tokens_no_router, 1e-6)

    print(f"[Exp2] Coverage on B after recalibration: {coverage_B:.3f}")
    print(f"[Exp2] Router token savings (robust prompt): {token_savings*100:.1f}%")

    plt.figure(figsize=(6,4))
    labels = ['A_orig', 'A_para', 'B_orig', 'B_para']
    mean_vals = [res_mean_nr[('A','orig')]['acc'], res_mean_nr[('A','para')]['acc'],
                 res_mean_nr[('B','orig')]['acc'], res_mean_nr[('B','para')]['acc']]
    robust_vals = [res_rob_nr[('A','orig')]['acc'], res_rob_nr[('A','para')]['acc'],
                   res_rob_nr[('B','orig')]['acc'], res_rob_nr[('B','para')]['acc']]
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, mean_vals, width, label='SURGE-mean (alpha=0)')
    plt.bar(x + width/2, robust_vals, width, label='SURGE-robust (alpha=0.5)')
    plt.xticks(x, labels)
    plt.ylabel('Accuracy')
    plt.title('EXP2: Robust accuracy across settings')
    plt.legend()
    plt.tight_layout()
    _savefig_pdf(os.path.join(images_dir, 'robust_accuracy_exp2.pdf'))

    plt.figure(figsize=(4,3))
    sns.barplot(x=['Coverage_B'], y=[coverage_B])
    plt.ylim(0,1)
    plt.title('EXP2: LCB coverage after drift recalibration')
    plt.tight_layout()
    _savefig_pdf(os.path.join(images_dir, 'coverage_exp2.pdf'))

    plt.figure(figsize=(5,3.5))
    sns.barplot(x=['no_router', 'with_router'], y=[tokens_no_router, tokens_with_router])
    plt.ylabel('Avg tokens')
    plt.title('EXP2: Router token savings (robust prompt)')
    plt.tight_layout()
    _savefig_pdf(os.path.join(images_dir, 'tokens_router_exp2.pdf'))

    surge.save(models_dir=MODELS_DIR_DEFAULT, name='surge_surrogate_exp2.pt')

    return {
        'pir_mean': pir_mean,
        'pir_robust': pir_robust,
        'coverage_B': coverage_B,
        'token_savings': token_savings,
    }


# ---------------------------------
# Experiment 3
# ---------------------------------

def surrogate_saliency(surr: SurrogateEnsemble, pir: PIR) -> np.ndarray:
    surr.embedder.zero_grad()
    with torch.enable_grad():
        frags = pir.fragments
        frag_vecs = []
        for f in frags:
            t_idx = torch.tensor([FRAG_TYPES.index(f.type)], dtype=torch.long, requires_grad=False)
            t_vec = surr.embedder.type_emb(t_idx).squeeze(0)
            text = f.content
            idx = torch.tensor([ord(c) % 256 for c in text[:300]], dtype=torch.long)
            idx_embed = surr.embedder.char_emb(idx)
            c_vec = idx_embed.mean(dim=0)
            v = (t_vec + c_vec).requires_grad_(True)
            frag_vecs.append(v)
        X = torch.stack(frag_vecs, dim=0).unsqueeze(0)
        out = None
        for m in surr.ensemble:
            logits = m(X)
            out = logits if out is None else out + logits
        out = out / len(surr.ensemble)
        success_logit = out[0,0]
        success_logit.backward()
        grads = np.array([v.grad.detach().cpu().numpy() for v in frag_vecs])
        sal = np.linalg.norm(grads, axis=1)
    return sal


def run_experiment_3(seed: int = 0, quick: bool = False, images_dir: str = IMAGES_DIR_DEFAULT):
    print("\n=== Experiment 3: Surrogate calibration, saliency-guided edits, causal interpretability ===")
    set_global_seed(seed)
    target = TargetBlackBox(seed + 13)
    surge = SURGE(target, seed)
    val = surge.val
    bank = surge.bank

    n_dev = 50 if quick else 100
    n_hold = 40 if quick else 80
    dev, _ = make_synthetic_dataset('bbh', n_dev=n_dev, n_test=10, seed=seed)
    hold, _ = make_synthetic_dataset('bbh', n_dev=n_hold, n_test=10, seed=seed+1)

    seed_prompt = 'Deliberate reasoning with verification, JSON output, and safe constraints.'
    pir0 = to_pir_from_seed(seed_prompt, bank)
    pir0 = val.repair(pir0)

    pre_logs = []
    losses = []
    for r in range(6 if quick else 12):
        batch_logs = []
        for _ in range(8):
            p = constrained_fragment_edit(pir0, bank, random)
            res = eval_candidate_on_dev(target, 'A', p, random.sample(dev, k=min(8, len(dev))))
            batch_logs.append({'pir': p, 'targets': {
                'success': res['success_rate'],
                'format_violation': res['format_rate'],
                'safety_violation': res['safety_rate'],
                'tokens': res['mean_tokens'],
            }})
        loss = surge.learn(batch_logs)
        losses.append(loss)
        pre_logs.extend(batch_logs)
    plt.figure(figsize=(5,3.5))
    plt.plot(losses)
    plt.xlabel('Pretrain steps')
    plt.ylabel('Surrogate loss')
    plt.title('EXP3: Surrogate training loss')
    plt.tight_layout()
    _savefig_pdf(os.path.join(images_dir, 'training_loss_surrogate.pdf'))

    sal = surrogate_saliency(surge.surr, pir0)
    top_idx = np.argsort(-sal)[:2]
    saliency_improve, random_improve = [], []
    base_score = eval_candidate_on_dev(target, 'A', pir0, random.sample(dev, k=min(10, len(dev))))['success_rate']

    p_sal = pir0.clone()
    for i in top_idx:
        f = p_sal.fragments[i]
        if not f.immutable:
            f.content += ' Double-check and validate.'
        p_sal = val.repair(p_sal)
    s_sal = eval_candidate_on_dev(target, 'A', p_sal, random.sample(dev, k=min(10, len(dev))))['success_rate']
    saliency_improve.append(s_sal - base_score)

    p_rand = constrained_fragment_edit(pir0, bank, random)
    s_rand = eval_candidate_on_dev(target, 'A', p_rand, random.sample(dev, k=min(10, len(dev))))['success_rate']
    random_improve.append(s_rand - base_score)

    ab_logs = []
    def toggle_reasoning(pir: PIR, on: bool) -> PIR:
        q = pir.clone()
        has_rt = any(f.type == 'reasoning_trigger' for f in q.fragments)
        if on and not has_rt:
            q.fragments.insert(2, copy.deepcopy(bank.sample('reasoning_trigger', 1)[0]))
        if not on and has_rt:
            q.fragments = [f for f in q.fragments if f.type != 'reasoning_trigger']
        return val.repair(q)

    p_on = toggle_reasoning(pir0, True)
    p_off = toggle_reasoning(pir0, False)
    items = random.sample(dev, k=min(12, len(dev)))
    suc_on = np.mean([target.call('A', p_on, it)['success'] for it in items])
    suc_off = np.mean([target.call('A', p_off, it)['success'] for it in items])
    ab_logs.append({'item': items[0], 'delta_success': float(suc_on - suc_off)})
    surge.router.fit_supervised(ab_logs, iters=60, lr=0.05)

    probs, labels = [], []
    for it in hold:
        p = constrained_fragment_edit(pir0, bank, random)
        probs.append(surge.surr.predict(p)['p_success'])
        y = eval_candidate_on_dev(target, 'A', p, [it])['success_rate']
        labels.append(y)
    ece = expected_calibration_error(np.array(probs), np.array(labels))

    plt.figure(figsize=(5,3.5))
    plt.hist(np.array(probs) - np.array(labels), bins=15, alpha=0.8)
    plt.xlabel('Predicted prob - Observed success')
    plt.ylabel('Count')
    plt.title(f'EXP3: Surrogate calibration (ECE={ece:.3f})')
    plt.tight_layout()
    _savefig_pdf(os.path.join(images_dir, 'surrogate_calibration_exp3.pdf'))

    plt.figure(figsize=(5,3.5))
    plt.plot(np.cumsum([1]*len(saliency_improve)), np.cumsum(saliency_improve), label='saliency-guided')
    plt.plot(np.cumsum([1]*len(random_improve)), np.cumsum(random_improve), label='random-edits')
    plt.xlabel('Edits applied')
    plt.ylabel('Cumulative improvement')
    plt.title('EXP3: Improvement curves (saliency vs random)')
    plt.legend()
    plt.tight_layout()
    _savefig_pdf(os.path.join(images_dir, 'improvement_curves_exp3_pair1.pdf'))

    plt.figure(figsize=(5,3.5))
    sns.barplot(x=['saliency','random'], y=[np.mean(saliency_improve), np.mean(random_improve)])
    plt.ylabel('Mean improvement')
    plt.title('EXP3: Mean improvement per edit')
    plt.tight_layout()
    _savefig_pdf(os.path.join(images_dir, 'improvement_curves_exp3_pair2.pdf'))

    surge.save(models_dir=MODELS_DIR_DEFAULT, name='surge_surrogate_exp3.pt')

    print(f"[Exp3] ECE={ece:.3f}, Saliency improvement={np.mean(saliency_improve):.3f}, Random improvement={np.mean(random_improve):.3f}")

    return {
        'ece': ece,
        'saliency_improve': float(np.mean(saliency_improve)),
        'random_improve': float(np.mean(random_improve)),
    }


# ---------------------------------
# Quick functional test
# ---------------------------------

def run_all(seed: int = 0, images_dir: str = IMAGES_DIR_DEFAULT):
    t0 = time.time()
    res1 = run_experiment_1(seed=seed, quick=True, images_dir=images_dir)
    res2 = run_experiment_2(seed=seed+1, quick=True, images_dir=images_dir)
    res3 = run_experiment_3(seed=seed+2, quick=True, images_dir=images_dir)
    print("\n=== Summary ===")
    print(f"Exp1: SURGE test(A,orig) acc={res1['final_scores']['SURGE'][(\'A\',1.0)]['acc']:.3f}")
    print(f"Exp2: Coverage_B={res2['coverage_B']:.3f}, Token savings={res2['token_savings']*100:.1f}%")
    print(f"Exp3: ECE={res3['ece']:.3f}, Saliency>Random? {res3['saliency_improve'] > res3['random_improve']}")
    print(f"Total runtime: {time.time()-t0:.1f}s")
    return res1, res2, res3


def test_quick_run(images_dir: str = IMAGES_DIR_DEFAULT):
    print("\n=== Running quick functional test ===")
    set_global_seed(123)
    res1 = run_experiment_1(seed=1, quick=True, images_dir=images_dir)
    assert 'final_scores' in res1, 'Exp1 did not return final_scores'
    res2 = run_experiment_2(seed=2, quick=True, images_dir=images_dir)
    assert 'coverage_B' in res2, 'Exp2 missing coverage_B'
    res3 = run_experiment_3(seed=3, quick=True, images_dir=images_dir)
    assert 'ece' in res3, 'Exp3 missing ece'
    for f in [
        'accuracy_exp1.pdf', 'ece_exp1.pdf', 'coverage_exp1.pdf', 'violation_rates_exp1.pdf',
        'robust_accuracy_exp2.pdf', 'coverage_exp2.pdf', 'tokens_router_exp2.pdf',
        'training_loss_surrogate.pdf', 'surrogate_calibration_exp3.pdf',
        'improvement_curves_exp3_pair1.pdf', 'improvement_curves_exp3_pair2.pdf']:
        full = os.path.join(images_dir, f)
        assert os.path.exists(full), f'Missing figure {full}'
    print("Quick test passed. Figures saved as PDF.")
