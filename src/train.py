import os
import math
import json
import time
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except Exception:
    HAS_CODECARBON = False
    class EmissionsTracker:
        def __init__(self, **kwargs):
            pass
        def start(self):
            return None
        def stop(self):
            return None

from .evaluate import evaluate_model
from .preprocess import SyntheticSequenceDataset, build_dataloaders

# Matplotlib defaults for publication-quality PDFs
plt.rcParams.update({
    'pdf.fonttype': 42,  # TrueType
    'ps.fonttype': 42,
    'figure.dpi': 300,
})

# ---------------------------- Utilities ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------------------------- Core Components ----------------------------

@dataclass
class CISample:
    ci_g_per_kwh: float
    pue: float

class CIStream:
    """Carbon intensity stream with uncertainty. If CSV missing, generates synthetic tri-region traces."""
    def __init__(self, csv_path: Optional[str] = None, hours: int = 48):
        self.df = None
        if csv_path and os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path)
                if 'hour' in self.df.columns:
                    self.df.rename(columns={'hour': 'hour_idx'}, inplace=True)
                if 'mean_ci_g_per_kwh' not in self.df.columns:
                    if 'mean' in self.df.columns:
                        self.df.rename(columns={'mean': 'mean_ci_g_per_kwh'}, inplace=True)
                    if 'std' in self.df.columns:
                        self.df.rename(columns={'std': 'std_ci_g_per_kwh'}, inplace=True)
                if 'pue' not in self.df.columns:
                    self.df['pue'] = 1.15
            except Exception:
                self.df = None
        if self.df is None:
            regions = ["Nordics", "WestUS", "CentralEU"]
            rows = []
            for h in range(hours):
                day_phase = 0.5 * (1 + math.sin(2 * math.pi * (h % 24) / 24.0))
                base = {"Nordics": 90.0, "WestUS": 300.0, "CentralEU": 450.0}
                stds = {"Nordics": 25.0, "WestUS": 110.0, "CentralEU": 160.0}
                pue_base = {"Nordics": 1.10, "WestUS": 1.18, "CentralEU": 1.22}
                for r in regions:
                    mean = max(10.0, base[r] * (0.8 + 0.4 * day_phase))
                    std = stds[r]
                    q10 = max(5.0, mean - 1.3 * std)
                    q50 = mean
                    q90 = mean + 1.3 * std
                    rows.append({
                        'hour_idx': h, 'region': r,
                        'mean_ci_g_per_kwh': mean, 'std_ci_g_per_kwh': std,
                        'q10': q10, 'q50': q50, 'q90': q90,
                        'pue': pue_base[r]
                    })
            self.df = pd.DataFrame(rows)
        self.regions = sorted(self.df['region'].unique())
        self.hours = int(self.df['hour_idx'].max()) + 1

    def sample(self, hour_idx: int, region: str, mode: str = 'normal') -> CISample:
        if region not in self.regions:
            region = self.regions[0]
        df_r = self.df[self.df['region'] == region]
        row = df_r[df_r['hour_idx'] == (hour_idx % len(df_r))]
        if row.empty:
            row = df_r.iloc[[hour_idx % len(df_r)]]
        r = row.iloc[0]
        if mode == 'quantile':
            u = np.random.rand()
            if u < 0.5:
                ci = np.interp(u, [0.1, 0.5], [r['q10'], r['q50']])
            else:
                ci = np.interp(u, [0.5, 0.9], [r['q50'], r['q90']])
        else:
            ci = max(0.0, np.random.normal(r['mean_ci_g_per_kwh'], r['std_ci_g_per_kwh']))
        return CISample(ci_g_per_kwh=float(ci), pue=float(r['pue']))

class EmbodiedTracker:
    def __init__(self, embodied_kg_per_gpu: float = 3000.0, lifetime_hours: float = 35000.0, wear_mult: float = 1.0):
        self.per_hour = embodied_kg_per_gpu / lifetime_hours
        self.wear_mult = wear_mult
    def attrib(self, gpu_hours: float, util: float = 0.9):
        return float(self.per_hour * gpu_hours * (0.5 + 0.5 * util) * self.wear_mult)

class NetworkCO2:
    def __init__(self, kg_per_gb: float = 0.0005):
        self.kg_per_gb = kg_per_gb
    def attrib(self, bytes_moved: int) -> float:
        return float((bytes_moved / (1024 ** 3)) * self.kg_per_gb)

class CPI:
    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha
    def cvar(self, samples: np.ndarray) -> float:
        if samples.size == 0:
            return 1e-6
        q = np.quantile(samples, self.alpha)
        tail = samples[samples >= q]
        return float(np.mean(tail)) if tail.size > 0 else float(np.mean(samples))
    def score(self, dprog: np.ndarray, co2: np.ndarray) -> float:
        return float(np.mean(dprog) / max(1e-6, self.cvar(co2)))

class DualBudget:
    def __init__(self, carbon_budget: float, deadline_hours: int, step: float = 0.05):
        self.remaining = float(carbon_budget)
        self.deadline = int(deadline_hours)
        self.lmbda = 0.0
        self.step = float(step)
    def update(self, realized_co2: float, hour: int):
        self.remaining -= realized_co2
        allowance = max(0.0, self.remaining) / max(1, (self.deadline - hour))
        slack = realized_co2 - allowance
        self.lmbda = max(0.0, self.lmbda + self.step * slack)

class DeltaEvals:
    """Online progress meta-model (simple EWMA + uncertainty)"""
    def __init__(self, init_mu: float = 0.02, init_sigma: float = 0.02, decay: float = 0.8):
        self.mu: Dict[str, float] = {}
        self.s2: Dict[str, float] = {}
        self.decay = decay
        self.init_mu = init_mu
        self.init_sigma = init_sigma
    def _key(self, recipe: Dict[str, Any]) -> str:
        return json.dumps({k: recipe[k] for k in sorted(recipe.keys())}, sort_keys=True)
    def update(self, recipe: Dict[str, Any], dloss_per_hour: float):
        k = self._key(recipe)
        mu = self.mu.get(k, self.init_mu)
        s2 = self.s2.get(k, self.init_sigma ** 2)
        new_mu = self.decay * mu + (1 - self.decay) * dloss_per_hour
        new_s2 = self.decay * s2 + (1 - self.decay) * (dloss_per_hour - mu) ** 2
        self.mu[k] = new_mu
        self.s2[k] = max(1e-6, new_s2)
    def sample_progress(self, recipe: Dict[str, Any], n: int = 64) -> np.ndarray:
        k = self._key(recipe)
        mu = self.mu.get(k, self.init_mu)
        sigma = math.sqrt(self.s2.get(k, self.init_sigma ** 2))
        return np.random.normal(mu, sigma, size=n)

# ---------------------------- Models and Trainer ----------------------------

class TinyLM(nn.Module):
    """A tiny causal LM: embedding -> GRU -> linear head."""
    def __init__(self, vocab_size=256, d_model=64, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, hidden, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.emb(input_ids)
        out, _ = self.rnn(x)
        out = self.ln(out)
        logits = self.fc(out)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss

class TrainerLite:
    def __init__(self, model: nn.Module, train_dl, val_dl, lr: float = 2e-3):
        self.model = model.to(DEVICE)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.global_step = 0

    def train_steps(self, steps: int, grad_accum: int = 1):
        self.model.train()
        it = iter(self.train_dl)
        for s in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_dl)
                batch = next(it)
            inp = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            _, loss = self.model(inp, labels=labels)
            loss.backward()
            if (s + 1) % grad_accum == 0:
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
            self.global_step += 1

    def evaluate(self) -> Tuple[float, float]:
        return evaluate_model(self.model, self.val_dl)

# ---------------------------- Experiment 1: Single-tenant CPI Controller ----------------------------

@dataclass
class Action:
    kind: str
    region: str
    duration_min: int
    recipe: Dict[str, Any]

class CPIController:
    def __init__(self, ci_stream: CIStream, delta_evals: DeltaEvals, cpi: CPI, dual_budget: DualBudget, regions: List[str]):
        self.ci = ci_stream
        self.de = delta_evals
        self.cpi = cpi
        self.db = dual_budget
        self.regions = regions

    def estimate_energy_kwh(self, action: Action) -> float:
        base_w = 220.0
        if action.recipe.get('power_cap_w') is not None:
            base_w = min(base_w, float(action.recipe['power_cap_w']))
        if action.kind == 'eval':
            base_w *= 0.5
        if action.kind == 'filler':
            base_w *= 0.25
        return float(base_w * (action.duration_min / 60.0) / 1000.0)

    def choose(self, hour_idx: int, candidates: List[Action]) -> Action:
        best, best_score = None, -1e18
        for a in candidates:
            rec = dict(a.recipe)
            rec['region'] = a.region
            dprog = self.de.sample_progress(rec, n=100) * (a.duration_min / 60.0)
            co2_samples = []
            for _ in range(100):
                cis = self.ci.sample(hour_idx, a.region)
                e_kwh = self.estimate_energy_kwh(a) * cis.pue
                co2_samples.append(e_kwh * (cis.ci_g_per_kwh / 1000.0))
            co2 = np.array(co2_samples)
            score = np.mean(dprog) / max(1e-6, (1.0 + self.db.lmbda) * self.cpi.cvar(co2))
            if score > best_score and np.mean(co2) <= max(0.0, self.db.remaining) + 2.0:
                best, best_score = a, score
        return best if best is not None else candidates[0]

# Baseline policies for Exp1

def baseline_fixedplan(hour_idx: int, candidates: List[Action]) -> Action:
    for a in candidates:
        if a.kind == 'train' and a.region == 'WestUS':
            return a
    return candidates[0]

def baseline_energy_only(hour_idx: int, candidates: List[Action]) -> Action:
    trains = [a for a in candidates if a.kind == 'train']
    if not trains:
        return candidates[0]
    def pc(a):
        return float(a.recipe.get('power_cap_w', 220.0))
    return sorted(trains, key=pc)[0]

def baseline_carbon_aware_placement(hour_idx: int, ci: CIStream, candidates: List[Action]) -> Action:
    best_r, best_ci = None, 1e9
    for r in ci.regions:
        c = ci.sample(hour_idx, r)
        if c.ci_g_per_kwh < best_ci:
            best_ci, best_r = c.ci_g_per_kwh, r
    trains = [a for a in candidates if a.kind == 'train' and a.region == best_r]
    if trains:
        return trains[0]
    return candidates[0]

def baseline_naive_pausing(hour_idx: int, ci: CIStream, candidates: List[Action], threshold_ci: float = 250.0) -> Action:
    best_r, best_ci = None, 1e9
    for r in ci.regions:
        c = ci.sample(hour_idx, r)
        if c.ci_g_per_kwh < best_ci:
            best_ci, best_r = c.ci_g_per_kwh, r
    if best_ci > threshold_ci:
        fills = [a for a in candidates if a.kind == 'filler']
        return fills[0] if fills else candidates[0]
    trains = [a for a in candidates if a.kind == 'train' and a.region == best_r]
    return trains[0] if trains else candidates[0]

# ---------------------------- Experiment 2: COSP + Multilevel + Fillers ----------------------------

class COSPPlanner:
    def __init__(self, ci_stream: CIStream, delta_evals: DeltaEvals, horizon_h: int = 24, regions: List[str] = None):
        self.ci = ci_stream
        self.de = delta_evals
        self.H = horizon_h
        self.regions = regions if regions is not None else self.ci.regions

    def energy_kwh(self, level: int, recipe: Dict[str, Any], duration_min: int = 60) -> float:
        base_w = 160.0 if level == 0 else 260.0
        if 'power_cap_w' in recipe:
            base_w = min(base_w, float(recipe['power_cap_w']))
        return float(base_w * (duration_min / 60.0) / 1000.0)

    def expected_co2(self, hour: int, region: str, level: int, recipe: Dict[str, Any], duration_min: int = 60) -> float:
        cis = self.ci.sample(hour, region, mode='quantile')
        return self.energy_kwh(level, recipe, duration_min) * cis.pue * (cis.ci_g_per_kwh / 1000.0)

    def expected_progress(self, level: int, recipe: Dict[str, Any], duration_min: int = 60) -> float:
        rec = dict(recipe)
        rec['level'] = level
        rec['region'] = 'plan'
        d = self.de.sample_progress(rec, n=64)
        return float(np.mean(d) * (duration_min / 60.0))

    def plan(self, start_hour: int, loss_target_reduction: float, levels=(0, 1)) -> List[Tuple[int, str, int, Dict[str, Any]]]:
        plan = []
        loss_left = loss_target_reduction
        for h in range(start_hour, start_hour + self.H):
            region_ci = [(r, self.ci.sample(h, r, mode='quantile').ci_g_per_kwh) for r in self.regions]
            region, min_ci = sorted(region_ci, key=lambda x: x[1])[0]
            if min_ci < 200.0 and 1 in levels:
                act = ('train', region, 1, {'precision': 'bf16', 'batch': 64, 'grad_accum': 2})
            elif min_ci < 350.0 and 0 in levels:
                act = ('train', region, 0, {'precision': 'bf16', 'batch': 64, 'grad_accum': 2, 'power_cap_w': 200})
            else:
                act = ('filler', region, -1, {'kind': 'dedup'})
            plan.append((h, *act))
            if act[0] == 'train':
                loss_left -= self.expected_progress(act[2], act[3])
            else:
                loss_left -= 0.01
            if loss_left <= 0:
                break
        return plan

class MultiLevelSystem:
    def __init__(self, train_dl, val_dl):
        self.model0 = TinyLM(vocab_size=256, d_model=48, hidden=48).to(DEVICE)
        self.model1 = TinyLM(vocab_size=256, d_model=80, hidden=80).to(DEVICE)
        self.opt0 = optim.Adam(self.model0.parameters(), lr=2e-3)
        self.opt1 = optim.Adam(self.model1.parameters(), lr=1.5e-3)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.it = iter(self.train_dl)

    def _next_batch(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.train_dl)
            return next(self.it)

    def train_level(self, level: int, steps: int, grad_accum: int = 2):
        model = self.model0 if level == 0 else self.model1
        opt = self.opt0 if level == 0 else self.opt1
        model.train()
        for s in range(steps):
            batch = self._next_batch()
            inp = batch['input_ids'].to(DEVICE)
            lab = batch['labels'].to(DEVICE)
            _, loss = model(inp, labels=lab)
            loss.backward()
            if (s + 1) % grad_accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def evaluate(self, level: int) -> Tuple[float, float]:
        model = self.model0 if level == 0 else self.model1
        return evaluate_model(model, self.val_dl)

    def distill_step(self, steps: int = 10, T: float = 2.0, alpha: float = 0.9):
        teacher = self.model1
        student = self.model0
        teacher.eval(); student.train()
        opt = self.opt0
        for _ in range(steps):
            batch = self._next_batch()
            inp = batch['input_ids'].to(DEVICE)
            lab = batch['labels'].to(DEVICE)
            with torch.no_grad():
                t_logits, _ = teacher(inp, labels=lab)
            s_logits, _ = student(inp, labels=lab)
            loss_kd = F.kl_div(F.log_softmax(s_logits / T, dim=-1), F.softmax(t_logits / T, dim=-1), reduction='batchmean') * (T * T)
            loss_hard = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), lab.reshape(-1))
            loss = alpha * loss_kd + (1 - alpha) * loss_hard
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)

# ---------------------------- Experiment 3: Multi-tenant ----------------------------

@dataclass
class Tenant:
    name: str
    target_prog: float
    carbon_budget: float
    deadline_h: int
    qual_coeff: float = 1.0
    remaining: float = None
    lambda_c: float = 0.0
    def __post_init__(self):
        self.remaining = float(self.target_prog)

class MultiTenantController:
    def __init__(self, tenants: List[Tenant], ci_stream: CIStream, regions: List[str], power_caps=(160,200,250)):
        self.tenants = tenants
        self.ci = ci_stream
        self.regions = regions
        self.power_caps = power_caps

    def allocate(self, hour: int, slots: int = 2) -> List[Tuple[Tenant, str, int, float, float]]:
        props = []
        for t in self.tenants:
            best = None
            best_score = -1e18
            for r in self.regions:
                ci = self.ci.sample(hour, r).ci_g_per_kwh
                for p in self.power_caps:
                    throughput = p / 250.0
                    dprog = max(0.0, throughput * t.qual_coeff * np.random.normal(1.0, 0.05))
                    co2 = (ci / 1000.0) * (p / 1000.0)
                    cpi = dprog / max(1e-6, co2)
                    score = cpi - t.lambda_c * co2
                    if score > best_score:
                        best_score = score
                        best = (t, r, p, dprog, co2)
            props.append(best)
        props_sorted = sorted(props, key=lambda x: (x[3]/max(1e-6,x[4])), reverse=True)
        return props_sorted[:slots]

    def step_dual(self, tenant: Tenant, realized_co2: float, eta: float = 0.05):
        allowance = max(0.0, tenant.carbon_budget) / max(1, tenant.deadline_h)
        slack = realized_co2 - allowance
        tenant.lambda_c = max(0.0, tenant.lambda_c + eta * slack)

# ---------------------------- Runner functions ----------------------------

def run_experiment1(img_dir: str, log_dir: str, models_dir: str, hours: int = 6, seed: int = 0, carbon_budget: float = 1.5,
                    steps_per_hour: int = 60, csv_trace: Optional[str] = None, save_prefix: str = "exp1") -> Dict[str, Any]:
    set_seed(seed)
    ensure_dir(img_dir); ensure_dir(log_dir); ensure_dir(models_dir)

    ds_train_A = SyntheticSequenceDataset(n_samples=1200, pattern='structured', seed=seed)
    ds_val_A   = SyntheticSequenceDataset(n_samples=200,  pattern='structured', seed=seed+1)
    ds_train_B = SyntheticSequenceDataset(n_samples=1200, pattern='noisy', seed=seed+2)
    ds_val_B   = SyntheticSequenceDataset(n_samples=200,  pattern='noisy', seed=seed+3)

    train_ds = torch.utils.data.ConcatDataset([ds_train_A, ds_train_B])
    val_ds   = torch.utils.data.ConcatDataset([ds_val_A, ds_val_B])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

    def init_model():
        m = TinyLM(vocab_size=256, d_model=64, hidden=64)
        return m
    base_state_dict = init_model().state_dict()

    def make_trainer_from_base():
        m = TinyLM(vocab_size=256, d_model=64, hidden=64)
        m.load_state_dict(base_state_dict)
        return TrainerLite(m, train_dl, val_dl, lr=2e-3)

    ci = CIStream(csv_path=csv_trace, hours=max(24, hours))
    regions = ci.regions
    de = DeltaEvals()
    cpi = CPI(alpha=0.8)
    db = DualBudget(carbon_budget=carbon_budget, deadline_hours=hours, step=0.2)
    controller = CPIController(ci, de, cpi, db, regions)

    def candidate_actions():
        return [
            Action('train', 'WestUS',   60, {'precision': 'bf16', 'batch': 64,  'grad_accum': 2}),
            Action('train', 'Nordics',  60, {'precision': 'bf16', 'batch': 64,  'grad_accum': 2, 'power_cap_w': 180}),
            Action('train', 'CentralEU',60, {'precision': 'fp16', 'batch': 48,  'grad_accum': 4}),
            Action('eval', 'WestUS',    15, {}),
            Action('filler', 'WestUS',  30, {'kind': 'dedup'})
        ]

    def exec_hour(trainer: TrainerLite, a: Action, hour_idx: int) -> Dict[str, Any]:
        tracker = EmissionsTracker(measure_power_secs=5, save_to_file=False) if HAS_CODECARBON else EmissionsTracker()
        tracker.start()
        if a.kind == 'train':
            grad_acc = int(a.recipe.get('grad_accum', 2))
            trainer.train_steps(steps=steps_per_hour, grad_accum=grad_acc)
        elif a.kind == 'eval':
            _ = trainer.evaluate()
        elif a.kind == 'filler':
            _ = sum([hash((hour_idx, i, a.region)) % 7919 for i in range(2000)])
        emissions = tracker.stop()
        cis = ci.sample(hour_idx, a.region)
        energy_kwh = controller.estimate_energy_kwh(a) * cis.pue
        co2_modeled = energy_kwh * (cis.ci_g_per_kwh / 1000.0)
        co2_measured = float(emissions) if emissions is not None else None
        return {
            'co2_modeled': co2_modeled,
            'co2_measured': co2_measured,
            'ci_g_per_kwh': cis.ci_g_per_kwh,
            'pue': cis.pue
        }

    # CAMLTO-Delta policy
    trainer_cam = make_trainer_from_base()
    loss_cam, acc_cam = trainer_cam.evaluate()
    hist_cam = []
    print(f"[Exp1] CAMLTO-Delta initial: loss={loss_cam:.4f}, acc={acc_cam:.4f}")
    for h in range(hours):
        cand = candidate_actions()
        chosen = controller.choose(h, cand)
        res = exec_hour(trainer_cam, chosen, h)
        new_loss, new_acc = trainer_cam.evaluate()
        d_loss = float(loss_cam - new_loss)
        rec = dict(chosen.recipe)
        rec['region'] = chosen.region
        if chosen.kind == 'train':
            de.update(rec, d_loss / max(1e-6, (chosen.duration_min / 60.0)))
        realized_co2 = res['co2_measured'] if res['co2_measured'] is not None else res['co2_modeled']
        db.update(realized_co2=realized_co2, hour=h)
        hist_cam.append({
            'hour': h, 'action': chosen.kind, 'region': chosen.region,
            'loss': new_loss, 'acc': new_acc,
            'dloss': d_loss, 'kgco2': realized_co2,
            'ci': res['ci_g_per_kwh'], 'pue': res['pue'],
            'lambda': db.lmbda, 'budget_remaining': db.remaining
        })
        loss_cam, acc_cam = new_loss, new_acc
        print(f"[Exp1][CAMLTO] hour={h:02d} act={chosen.kind:<6} reg={chosen.region:<9} loss={loss_cam:.4f} acc={acc_cam:.4f} co2={realized_co2:.4f} kg, lambda={db.lmbda:.3f}, budget_rem={db.remaining:.3f}")

    df_cam = pd.DataFrame(hist_cam)
    cam_csv = os.path.join(log_dir, f"{save_prefix}_camltodelta_log.csv")
    df_cam.to_csv(cam_csv, index=False)

    # Baselines
    def run_baseline(policy_name: str, chooser_fn) -> pd.DataFrame:
        trainer = make_trainer_from_base()
        loss, acc = trainer.evaluate()
        hist = []
        for h in range(hours):
            cand = candidate_actions()
            a = chooser_fn(h, ci, cand) if 'ci' in chooser_fn.__code__.co_varnames else chooser_fn(h, cand)
            res = exec_hour(trainer, a, h)
            new_loss, new_acc = trainer.evaluate()
            hist.append({
                'hour': h, 'action': a.kind, 'region': a.region,
                'loss': new_loss, 'acc': new_acc,
                'dloss': loss - new_loss,
                'kgco2': res['co2_measured'] if res['co2_measured'] is not None else res['co2_modeled'],
                'ci': res['ci_g_per_kwh'], 'pue': res['pue']
            })
            loss, acc = new_loss, new_acc
            print(f"[Exp1][{policy_name}] hour={h:02d} act={a.kind:<6} reg={a.region:<9} loss={loss:.4f} acc={acc:.4f} co2={hist[-1]['kgco2']:.4f} kg")
        df = pd.DataFrame(hist)
        df.to_csv(os.path.join(log_dir, f"{save_prefix}_{policy_name.lower()}_log.csv"), index=False)
        return df

    df_fixed = run_baseline("FixedPlan", baseline_fixedplan)
    df_energy = run_baseline("EnergyOnly", baseline_energy_only)
    df_capl = run_baseline("CarbonAwarePlacement", baseline_carbon_aware_placement)
    df_pause = run_baseline("NaivePausing", baseline_naive_pausing)

    # Plots
    sns.set_style("whitegrid")

    plt.figure(figsize=(6,4))
    plt.plot(df_cam['hour'], df_cam['loss'], label='CAMLTO-Delta')
    plt.plot(df_fixed['hour'], df_fixed['loss'], label='FixedPlan')
    plt.plot(df_energy['hour'], df_energy['loss'], label='EnergyOnly')
    plt.plot(df_capl['hour'], df_capl['loss'], label='CarbonAwarePlacement')
    plt.plot(df_pause['hour'], df_pause['loss'], label='NaivePausing')
    plt.xlabel('Hour'); plt.ylabel('Validation Loss'); plt.title('Validation Loss vs Time'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(img_dir, "training_loss_camltodelta.pdf"), bbox_inches="tight"); plt.close()

    def cpi_df(df):
        eps=1e-6
        return (df['dloss'] / (df['kgco2'] + eps)).replace(np.inf, np.nan).fillna(0)
    plt.figure(figsize=(6,4))
    plt.plot(df_cam['hour'], cpi_df(df_cam), label='CAMLTO-Delta')
    plt.xlabel('Hour'); plt.ylabel('Δloss per kgCO2'); plt.title('CPI Trajectory (Hourly)')
    plt.tight_layout(); plt.savefig(os.path.join(img_dir, "cpi_camltodelta.pdf"), bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(df_cam['hour'], df_cam['kgco2'].cumsum(), label='CAMLTO-Delta')
    plt.plot(df_fixed['hour'], df_fixed['kgco2'].cumsum(), label='FixedPlan')
    plt.plot(df_energy['hour'], df_energy['kgco2'].cumsum(), label='EnergyOnly')
    plt.plot(df_capl['hour'], df_capl['kgco2'].cumsum(), label='CarbonAwarePlacement')
    plt.plot(df_pause['hour'], df_pause['kgco2'].cumsum(), label='NaivePausing')
    plt.xlabel('Hour'); plt.ylabel('Cumulative kgCO2'); plt.title('kgCO2 vs Time'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(img_dir, "kgco2_vs_time_camltodelta.pdf"), bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(df_cam['hour'], df_cam['budget_remaining'])
    plt.xlabel('Hour'); plt.ylabel('Remaining Budget (kgCO2)'); plt.title('Carbon Budget Consumption (CAMLTO-Delta)')
    plt.tight_layout(); plt.savefig(os.path.join(img_dir, "budget_camltodelta.pdf"), bbox_inches="tight"); plt.close()

    # Save final model
    torch.save(trainer_cam.model.state_dict(), os.path.join(models_dir, f"{save_prefix}_final_model.pt"))

    def summarize(df, name):
        return {
            'final_loss': float(df['loss'].iloc[-1]),
            'cumu_co2': float(df['kgco2'].sum()),
            'mean_hourly_cpi': float((df['dloss'] / (df['kgco2'] + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0).mean()),
            'name': name
        }

    summary = [
        summarize(df_cam, 'CAMLTO-Delta'),
        summarize(df_fixed, 'FixedPlan'),
        summarize(df_energy, 'EnergyOnly'),
        summarize(df_capl, 'CarbonAwarePlacement'),
        summarize(df_pause, 'NaivePausing')
    ]
    for s in summary:
        print(f"[Exp1][Summary] {s['name']:<22} loss={s['final_loss']:.4f} cumCO2={s['cumu_co2']:.4f} meanCPI={s['mean_hourly_cpi']:.5f}")

    return {
        'cam': df_cam,
        'fixed': df_fixed,
        'energy': df_energy,
        'capl': df_capl,
        'pause': df_pause,
        'summary': summary
    }


def run_experiment2(img_dir: str, log_dir: str, models_dir: str, hours: int = 8, seed: int = 1, csv_trace: Optional[str] = None, steps_per_hour: int = 60,
                    save_prefix: str = "exp2") -> Dict[str, Any]:
    set_seed(seed)
    ensure_dir(img_dir); ensure_dir(log_dir); ensure_dir(models_dir)

    train_dl, val_dl = build_dataloaders(seed=seed, train_n=1500, val_n=300, batch_size_train=32, batch_size_val=64)

    sys_ml = MultiLevelSystem(train_dl, val_dl)
    ci = CIStream(csv_path=csv_trace, hours=max(24, hours))
    regions = ci.regions
    de = DeltaEvals()
    planner = COSPPlanner(ci, de, horizon_h=hours, regions=regions)

    target_reduction = 0.25
    plan = planner.plan(0, target_reduction, levels=(0,1))

    hist = []
    embodied = EmbodiedTracker()
    net_co2 = NetworkCO2()

    for idx, (h, kind, region, level, recipe) in enumerate(plan):
        if h >= hours:
            break
        if kind == 'train':
            sys_ml.train_level(level, steps=steps_per_hour, grad_accum=int(recipe.get('grad_accum', 2)))
            loss, acc = sys_ml.evaluate(level)
            observed_dloss = 0.02
            de.update({'precision':recipe.get('precision','bf16'), 'batch':recipe.get('batch',64),
                       'grad_accum':recipe.get('grad_accum',2), 'power_cap_w':recipe.get('power_cap_w',220),
                       'region':region, 'level':level}, observed_dloss)
        elif kind == 'filler':
            _ = sum([hash((h, i, region)) % 7919 for i in range(3000)])
            sys_ml.distill_step(steps=20)
            loss, acc = sys_ml.evaluate(level=0)
        else:
            loss, acc = sys_ml.evaluate(level=0)

        duration_min = 60 if kind != 'filler' else 30
        if kind == 'train':
            e_kwh = planner.energy_kwh(level, recipe, duration_min)
        else:
            e_kwh = planner.energy_kwh(level=0, recipe={'power_cap_w':160}, duration_min=duration_min) * 0.2
        cis = ci.sample(h, region)
        op_co2 = e_kwh * cis.pue * (cis.ci_g_per_kwh / 1000.0)
        emb_co2 = embodied.attrib(gpu_hours=(duration_min/60.0), util=0.6 if kind=='filler' else 0.9)
        wan_co2 = net_co2.attrib(bytes_moved=50 * 1024**2 if kind=='filler' else 200 * 1024**2)
        tot_co2 = op_co2 + emb_co2 + wan_co2

        hist.append({
            'hour': h, 'kind': kind, 'region': region, 'level': level,
            'loss': loss, 'acc': acc, 'op_co2': op_co2, 'emb_co2': emb_co2, 'wan_co2': wan_co2, 'tot_co2': tot_co2
        })
        print(f"[Exp2] h={h:02d} kind={kind:<6} reg={region:<9} lvl={level} loss={loss:.4f} acc={acc:.4f} co2_tot={tot_co2:.4f} (op={op_co2:.4f}, emb={emb_co2:.4f}, wan={wan_co2:.6f})")

    df = pd.DataFrame(hist).sort_values('hour')
    df.to_csv(os.path.join(log_dir, f"{save_prefix}_cosp_log.csv"), index=False)

    sns.set_style("whitegrid")
    plt.figure(figsize=(6,4))
    plt.plot(df['hour'], df['loss'], marker='o')
    plt.xlabel('Hour'); plt.ylabel('Validation Loss'); plt.title('Validation Loss with COSP + Multilevel + Fillers')
    plt.tight_layout(); plt.savefig(os.path.join(img_dir, "training_loss_cosp.pdf"), bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(df['hour'], df['tot_co2'].cumsum(), label='Total')
    plt.plot(df['hour'], df['op_co2'].cumsum(), label='Operational')
    plt.plot(df['hour'], df['emb_co2'].cumsum(), label='Embodied')
    plt.plot(df['hour'], df['wan_co2'].cumsum(), label='WAN')
    plt.xlabel('Hour'); plt.ylabel('Cumulative kgCO2'); plt.title('kgCO2 vs Time (COSP)'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(img_dir, "kgco2_vs_time_cosp.pdf"), bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6,4))
    frac_fill = (df['kind'] == 'filler').astype(float)
    plt.bar(df['hour'], frac_fill)
    plt.xlabel('Hour'); plt.ylabel('Filler Indicator'); plt.title('Filler Windows (COSP Plan)')
    plt.tight_layout(); plt.savefig(os.path.join(img_dir, "time_in_fillers_cosp.pdf"), bbox_inches="tight"); plt.close()

    print(f"[Exp2][Summary] Final loss={df['loss'].iloc[-1]:.4f} total kgCO2={df['tot_co2'].sum():.4f}")
    return {'df': df}


def cfe_alignment(logs_df: pd.DataFrame, clean_supply_df: pd.DataFrame) -> float:
    df = logs_df.merge(clean_supply_df, on=['hour', 'region'], how='left')
    df['clean_frac'] = df['clean_frac'].fillna(0.0)
    df['clean_matched'] = np.minimum(df['energy_kwh'], df['energy_kwh'] * df['clean_frac'])
    return float(df['clean_matched'].sum() / max(1e-6, df['energy_kwh'].sum()))


def run_experiment3(img_dir: str, log_dir: str, models_dir: str, hours: int = 24, seed: int = 2, slots: int = 2, csv_trace: Optional[str] = None,
                    save_prefix: str = "exp3") -> Dict[str, Any]:
    set_seed(seed)
    ensure_dir(img_dir); ensure_dir(log_dir); ensure_dir(models_dir)

    ci = CIStream(csv_path=csv_trace, hours=max(24, hours))
    regions = ci.regions

    tenants = [
        Tenant("T1_pretrain", target_prog=5.0, carbon_budget=1.8, deadline_h=hours, qual_coeff=1.0),
        Tenant("T2_SFT",      target_prog=3.0, carbon_budget=1.2, deadline_h=hours, qual_coeff=0.9),
        Tenant("T3_RLHF",     target_prog=2.0, carbon_budget=0.9, deadline_h=hours, qual_coeff=0.8),
    ]

    ctrl = MultiTenantController(tenants, ci, regions)

    clean_rows = []
    for h in range(hours):
        for r in regions:
            base = 0.7 if r == 'Nordics' else 0.4 if r == 'WestUS' else 0.3
            frac = max(0.0, min(1.0, base + 0.2 * math.sin(2 * math.pi * (h % 24) / 24.0)))
            clean_rows.append({'hour': h, 'region': r, 'clean_frac': frac})
    clean_df = pd.DataFrame(clean_rows)

    logs = []
    for h in range(hours):
        allocs = ctrl.allocate(h, slots=slots)
        for (t, region, power, dprog, co2_model) in allocs:
            cis = ci.sample(h, region)
            energy_kwh = power / 1000.0
            co2 = energy_kwh * cis.pue * (cis.ci_g_per_kwh / 1000.0)
            noise = np.random.normal(0, 0.05 * dprog)
            t.remaining = max(0.0, t.remaining - max(0.0, dprog + noise))
            t.carbon_budget -= co2
            ctrl.step_dual(t, co2)
            logs.append({
                'hour': h, 'tenant': t.name, 'region': region, 'power': power,
                'energy_kwh': energy_kwh, 'co2': co2, 'remaining': t.remaining,
                'lambda': t.lambda_c
            })

    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(log_dir, f"{save_prefix}_multitenant_log.csv"), index=False)

    utilities = {}
    for t in tenants:
        achieved = max(0.0, t.target_prog - t.remaining)
        utilities[t.name] = achieved / max(1e-6, t.target_prog)
    util_series = pd.Series(utilities)
    jain = (util_series.sum() ** 2) / (len(util_series) * ((util_series ** 2).sum() + 1e-9))

    cfe = cfe_alignment(df.groupby(['hour','region']).agg({'energy_kwh':'sum'}).reset_index(), clean_df)

    budget_violations = {t.name: float(-min(0.0, t.carbon_budget)) for t in tenants}

    print(f"[Exp3][Summary] Jain fairness={jain:.4f}, CFE alignment={cfe:.4f}")
    for t in tenants:
        print(f"[Exp3][Tenant] {t.name}: remaining_prog={t.remaining:.3f}, budget_left={t.carbon_budget:.3f}, lambda={t.lambda_c:.3f}")

    sns.set_style("whitegrid")

    plt.figure(figsize=(6,4))
    plt.bar(util_series.index, util_series.values)
    plt.xlabel('Tenant'); plt.ylabel('Normalized Utility (progress)'); plt.title(f'Utilities; Jain={jain:.3f}')
    plt.tight_layout(); plt.savefig(os.path.join(img_dir, "fairness_index.pdf"), bbox_inches="tight"); plt.close()

    energy_by_hour = df.groupby(['hour','region']).agg({'energy_kwh':'sum'}).reset_index()
    pivot_energy = energy_by_hour.pivot(index='hour', columns='region', values='energy_kwh').fillna(0.0)
    plt.figure(figsize=(7,4))
    pivot_energy.plot(kind='area', stacked=True, ax=plt.gca())
    plt.xlabel('Hour'); plt.ylabel('Energy (kWh)'); plt.title(f'Allocated Load by Region; CFE align={cfe:.3f}')
    plt.tight_layout(); plt.savefig(os.path.join(img_dir, "cfe_alignment.pdf"), bbox_inches="tight"); plt.close()

    bud = df.copy()
    init_budgets = {}
    for t in tenants:
        init_budgets[t.name] = float(t.carbon_budget + df[df['tenant']==t.name]['co2'].sum())
    bud['cum_co2'] = bud.groupby('tenant')['co2'].cumsum()
    bud['budget_rem'] = bud.apply(lambda r: init_budgets[r['tenant']] - r['cum_co2'], axis=1)
    plt.figure(figsize=(7,4))
    for name, g in bud.groupby('tenant'):
        plt.plot(g['hour'], g['budget_rem'], label=name)
    plt.xlabel('Hour'); plt.ylabel('Remaining Budget (kgCO2)'); plt.title('Budget Consumption per Tenant'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(img_dir, "budget_consumption_tenants.pdf"), bbox_inches="tight"); plt.close()

    return {
        'df': df,
        'jain': jain,
        'cfe': cfe,
        'budget_violations': budget_violations
    }
