import os
import math
import time
import json
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure headless plotting if needed by other modules that import this file (evaluate will set backend)

# Optional dependencies with fallbacks
try:
    from tdigest import TDigest
    TDIGEST_AVAILABLE = True
except Exception:
    TDIGEST_AVAILABLE = False

try:
    from confseq import bern_cs
    CONFSEQ_AVAILABLE = True
except Exception:
    CONFSEQ_AVAILABLE = False

try:
    from river import drift
    RIVER_AVAILABLE = True
except Exception:
    RIVER_AVAILABLE = False


# -----------------------------
# Utility: robust scale and quantile cap
# -----------------------------
class RollingMAD:
    def __init__(self, win=1024):
        self.win = int(win)
        self.buf = deque(maxlen=self.win)
    def update(self, x: float):
        self.buf.append(float(x))
    @property
    def value(self) -> float:
        if len(self.buf) == 0:
            return 1.0
        arr = np.fromiter(self.buf, dtype=float)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        return max(1e-6, 1.4826 * mad)


class QuantileCap:
    """Robust upper quantile cap with tdigest if available; else rolling reservoir.
    """
    def __init__(self, reservoir=5000, q_level: float = 0.995):
        self.reservoir = int(reservoir)
        self.deque = deque(maxlen=self.reservoir)
        self._td = TDigest() if TDIGEST_AVAILABLE else None
        self.q_level = float(q_level)
    def update(self, x: float):
        if self._td is not None:
            try:
                self._td.update(float(x))
            except Exception:
                pass
        self.deque.append(float(x))
    def q(self, default_scale: float) -> float:
        p = self.q_level
        if self._td is not None and len(self.deque) > 0:
            try:
                return float(self._td.quantile(p))
            except Exception:
                pass
        if len(self.deque) == 0:
            return 10.0 * max(1e-6, default_scale)
        arr = np.fromiter(self.deque, dtype=float)
        return float(np.quantile(arr, p))


# -----------------------------
# SAMC2 Core Components
# -----------------------------
@dataclass
class SAMC2Config:
    alpha: float = 0.1
    scales: tuple = tuple([2**i for i in range(0, 10)])  # 1..512
    gamma: float = 1.0
    c_I: float = 0.25
    eta0_mult: float = 0.06
    huber_delta: float = 0.5
    barrier_beta: float = 1.0
    lam: float = 0.7
    kappa: float = 0.5
    mu_cap: float = 5.0
    freeze_tau: int = 5
    warmup: int = 1000


class MultiResController:
    def __init__(self, cfg: SAMC2Config):
        self.cfg = cfg
        self.S = len(cfg.scales)
        self.scales = torch.tensor(cfg.scales, dtype=torch.float32)
        self.lambda_s = torch.exp(-1.0 / self.scales)
        self.lambda2_s = torch.exp(-2.0 / self.scales)
        self.E = torch.zeros(self.S)
        self.V = torch.zeros(self.S)
        self.qP = torch.zeros(self.S)
        self.I = torch.zeros(self.S)
        self.D = torch.zeros(self.S)
        self.sat = torch.full((self.S,), 3.0)  # multiplier of S_t
        self.eta_base = torch.tensor([cfg.eta0_mult/np.sqrt(1.0+w) for w in cfg.scales], dtype=torch.float32)
        self.huber_delta = cfg.huber_delta
        self.cap = QuantileCap(reservoir=8000)
        self.mad = RollingMAD(win=1024)
        self.S_t = 1.0
        self.last_proposals = torch.zeros(self.S)
        self._freeze_relax_timer = torch.zeros(self.S, dtype=torch.int32)

    @staticmethod
    def _huber(r: torch.Tensor, delta: float) -> torch.Tensor:
        return torch.where(torch.abs(r) <= delta, r, delta * torch.sign(r))

    def _update_stats(self, score: float):
        self.mad.update(score)
        self.S_t = self.mad.value
        self.cap.update(score)

    def q_max(self) -> float:
        q = self.cap.q(self.S_t)
        return max(10.0 * 1e-6, q)

    def update_after_observation(self, err: int, score: float):
        self._update_stats(score)
        alpha = self.cfg.alpha
        # EW updates for E,V (kernelized trackers)
        self.E = self.lambda_s * self.E + (err - alpha)
        self.V = self.lambda2_s * self.V + err * (1 - err)
        # I-term with saturation scaled by S_t
        sat_val = self.sat * self.S_t
        I_raw = self.cfg.c_I * self.S_t * torch.tanh(self.cfg.gamma * self.E / torch.sqrt(self.V + 1e-6))
        self.I = torch.clamp(I_raw, -sat_val, sat_val)
        # P-term: robust coverage-based Huber update
        eta = self.eta_base * self.S_t
        grad = self._huber(torch.tensor(float(err - alpha)), self.huber_delta)
        self.qP = torch.relu(self.qP + eta * grad)
        # D-term disabled by default
        self.D.zero_()
        # Proposal and barrier-clamped
        proposals = torch.relu(self.qP + self.I + self.D)
        qmax = self.q_max()
        # clamp proposals softly by qmax
        proposals = torch.clamp(proposals, 0.0, qmax * 1.2)
        self.last_proposals = proposals
        return proposals.detach().clone(), qmax


class SleepingExpertsAggregator:
    def __init__(self, cfg: SAMC2Config, num_scales: int):
        self.cfg = cfg
        self.S = int(num_scales)
        self.w = torch.full((self.S,), 1.0/self.S)
        self.cum_loss = torch.zeros(self.S)
        self.Z_t = 1.0
        self.eta = 0.5  # exp-weights step size
        self.last_losses = torch.zeros(self.S)
    def update_scale(self, scores_recent: List[float]):
        if len(scores_recent) > 0:
            arr = np.asarray(scores_recent, dtype=float)
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            self.Z_t = max(1e-6, 1.4826 * mad)
    def step(self, proposals: torch.Tensor, err: int, qmax: float, cs_pressures: torch.Tensor = None):
        lam = self.cfg.lam
        beta = self.cfg.barrier_beta
        widths = proposals
        # Barrier: discourage exceeding qmax
        barrier = torch.clamp((proposals - qmax) / max(qmax, 1e-6), min=0) ** 2
        loss = lam * abs(err - self.cfg.alpha) + (1 - lam) * (widths / max(self.Z_t, 1e-6)) + beta * barrier
        self.last_losses = loss.detach().clone()
        self.cum_loss += loss
        logits = -self.eta * self.cum_loss
        if cs_pressures is not None:
            logits = logits + self.cfg.kappa * cs_pressures
        self.w = torch.softmax(logits, dim=0)
        q = float(torch.dot(self.w, proposals))
        return q, loss.detach().clone()


# Dyadic CS Layer (fallback to simple heuristic if confseq is unavailable)
class DyadicCSLayer:
    def __init__(self, alpha: float, T_hint: int = 100_000, mu_cap: float = 5.0):
        self.alpha = float(alpha)
        self.t = 0
        self.levels = int(np.ceil(np.log2(max(2, T_hint))))
        self.mu = {}  # duals per interval key=(start, end)
        self.mu_cap = float(mu_cap)
        self.alarm = False
        self.n = 0
        self.sum_err = 0.0
        if CONFSEQ_AVAILABLE:
            self.cs = bern_cs.BernoulliCS()
        else:
            self.cs = None

    def intervals_covering_t(self, t: int):
        cov = []
        for Lpow in range(self.levels + 1):
            L = 2 ** Lpow
            start = (t // L) * L
            cov.append((start, start + L - 1, L))
        return cov

    def update(self, t: int, err: int, alpha: float):
        self.t = int(t)
        self.n += 1
        self.sum_err += float(err)
        cov = self.intervals_covering_t(t)
        for a, b, L in cov:
            key = (a, b)
            if key not in self.mu:
                self.mu[key] = 0.0
            rho = 1.0 / math.sqrt(max(1.0, float(L)))
            self.mu[key] = min(self.mu_cap, max(0.0, self.mu[key] + rho * (err - alpha)))
        # Alarm diagnostic
        if CONFSEQ_AVAILABLE and self.cs is not None:
            self.cs.add(int(err))
            lcb, ucb = self.cs.confidence_sequence(confidence=1 - 0.01)
            self.alarm = (lcb > alpha) or (ucb < alpha)
        else:
            if self.n > 0:
                phat = self.sum_err / self.n
                eps = math.sqrt(math.log(2.0 / 0.01) / (2 * max(1, self.n)))
                self.alarm = (phat - alpha > eps) or (alpha - phat > eps)

    def pressures_for_scales(self, t: int, scales_vec: np.ndarray) -> torch.Tensor:
        press = []
        for s in scales_vec:
            s = float(s)
            w = 0.0
            for (a, b), mu in self.mu.items():
                if a <= t <= b:
                    L = (b - a + 1)
                    w += mu * min(1.0, L / (L + s))
            press.append(w)
        return torch.tensor(press, dtype=torch.float32)


# -----------------------------
# Baselines
# -----------------------------
class PIDController:
    def __init__(self, alpha: float, c_I: float = 0.25, single_scale: float = 128.0):
        self.alpha = float(alpha)
        self.c_I = float(c_I)
        self.lambda1 = math.exp(-1.0 / float(single_scale))
        self.lambda2 = math.exp(-2.0 / float(single_scale))
        self.E = 0.0
        self.V = 0.0
        self.qP = 0.0
        self.gamma = 1.0
        self.huber_delta = 0.5
        self.eta = 0.06
        self.mad = RollingMAD(1024)
        self.cap = QuantileCap(reservoir=8000)
    def q_max(self) -> float:
        return self.cap.q(self.mad.value)
    def update(self, err: int, score: float) -> float:
        self.mad.update(score)
        self.cap.update(score)
        self.E = self.lambda1 * self.E + (err - self.alpha)
        self.V = self.lambda2 * self.V + err * (1 - err)
        S_t = self.mad.value
        I = self.c_I * S_t * math.tanh(self.gamma * self.E / math.sqrt(self.V + 1e-6))
        grad = max(-self.huber_delta, min(self.huber_delta, err - self.alpha))
        self.qP = max(0.0, self.qP + self.eta * S_t * grad)
        q = max(0.0, min(self.qP + I, 1.2 * self.q_max()))
        return float(q)


class RSC:
    def __init__(self, alpha: float, W_cal: int):
        self.alpha = float(alpha)
        self.W = int(W_cal)
        self.deque = deque(maxlen=self.W)
        self.ready = False
    def update(self, score: float):
        self.deque.append(float(score))
        if len(self.deque) >= max(32, int(0.2 * self.W)):
            self.ready = True
    def propose(self) -> float:
        if len(self.deque) == 0:
            return 1.0
        if not self.ready:
            return float(np.median(np.fromiter(self.deque, dtype=float)))
        q = float(np.quantile(np.fromiter(self.deque, dtype=float), 1 - self.alpha))
        return q


class CUSUMResetACI(PIDController):
    def __init__(self, alpha: float, c_I: float = 0.25, single_scale: float = 128.0, reset_threshold: float = 3.0):
        super().__init__(alpha, c_I, single_scale)
        self.reset_threshold = float(reset_threshold)
        if RIVER_AVAILABLE:
            self.detector = drift.ADWIN()
        else:
            # simple Page-Hinkley on squared residuals fallback
            self._ph_mean = 0.0
            self._ph_cum = 0.0
            self._ph_lambda = 50.0  # threshold
            self._ph_alpha = 0.999  # forgetting
    def _fallback_change_detect(self, z2: float) -> bool:
        self._ph_mean = self._ph_alpha * self._ph_mean + (1 - self._ph_alpha) * z2
        self._ph_cum = min(0.0, self._ph_cum + (z2 - self._ph_mean - 0.0))
        return abs(self._ph_cum) > self._ph_lambda
    def update(self, err: int, score: float) -> float:
        in_drift = False
        if RIVER_AVAILABLE:
            in_drift, _ = self.detector.update(score ** 2)
        else:
            in_drift = self._fallback_change_detect(score ** 2)
        if in_drift:
            # reset state
            self.E = 0.0
            self.V = 0.0
            self.qP = 0.0
            self.cap = QuantileCap(reservoir=8000)
        return super().update(err, score)


# -----------------------------
# Base Predictor
# -----------------------------
class SeasonalNaiveEWMA:
    def __init__(self, season_lag=24, bias_lag=169, bias_decay=0.98, bias_gain=0.02):
        self.y_hist = []
        self.b = 0.0
        self.season_lag = int(season_lag)
        self.bias_lag = int(bias_lag)
        self.bias_decay = float(bias_decay)
        self.bias_gain = float(bias_gain)
    def step(self, t: int, y_prev: float):
        if t >= self.season_lag:
            season_hat = self.y_hist[t - self.season_lag]
        else:
            season_hat = self.y_hist[-1] if self.y_hist else 0.0
        if len(self.y_hist) >= self.bias_lag:
            self.b = self.bias_decay * self.b + self.bias_gain * (self.y_hist[-1] - self.y_hist[-self.bias_lag])
        yhat = season_hat + self.b
        self.y_hist.append(float(y_prev))
        return float(yhat)


# -----------------------------
# Data Generators (Exp1 and Exp3)
# -----------------------------

def _t_noise(df, sigma, size, rng):
    z = rng.standard_t(df, size=size)
    return z * (sigma / math.sqrt(df / (df - 2)))


def generate_synth_ex1(T=50_000, seed=123):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    S = 2 * np.sin(2 * np.pi * np.arange(T) / 24.0) + 1.5 * np.sin(2 * np.pi * np.arange(T) / 168.0)
    eps = np.zeros(T)
    for t in range(T):
        if t < 10_000:
            eps[t] = _t_noise(5, 1.0, 1, rng)[0]
        elif t < 20_000:
            eps[t] = _t_noise(3, 1.2, 1, rng)[0] + 5.0
        elif t < 30_000:
            eps[t] = _t_noise(8, 0.6, 1, rng)[0]
        elif t < 40_000:
            if rng.random() < 0.1:
                eps[t] = rng.normal(0.0, 4.0)
            else:
                eps[t] = _t_noise(5, 1.0, 1, rng)[0]
        else:
            eps[t] = _t_noise(3, 2.0, 1, rng)[0]
        if t > 0:
            x[t] = 0.6 * x[t - 1] + rng.normal(0.0, 0.5)
        if 30_000 <= t < 40_000:
            x[t] += 0.01 * (t - 30_000)
    y = x + S + eps
    mask = rng.random(T) < 0.01
    y[mask] += np.sign(S[mask] + 1e-6) * 10.0 * np.abs(S[mask])
    return y.astype(float), S.astype(float)


def generate_regime_carousel_ex3(T=60_000, seed=7):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    segments = []
    t = 0
    W_star_seq = []  # optimal scales labels among {8, 32, 128, 512}
    while t < T:
        for kind in ["highfreq", "medium", "heavy", "stationary"]:
            if t >= T:
                break
            seg_len = int(rng.integers(5000, 10001))
            seg_end = min(T, t + seg_len)
            if kind == "highfreq":
                for i in range(t, seg_end):
                    y[i] = (y[i-1] if i > 0 else 0.0) + rng.normal(0.0, 0.5)
                W_star_seq.append((t, seg_end - 1, 8))
            elif kind == "medium":
                jump_idx = t + int(rng.integers(200, max(201, seg_len)))
                jump_A = float(rng.uniform(3.0, 7.0))
                for i in range(t, seg_end):
                    y[i] = (y[i-1] if i > 0 else 0.0) + rng.normal(0.0, 0.3)
                    if i == jump_idx and i < seg_end:
                        y[i:] += jump_A
                W_star_seq.append((t, seg_end - 1, 32))
            elif kind == "heavy":
                for i in range(t, seg_end):
                    y[i] = (y[i-1] if i > 0 else 0.0) + _t_noise(3, 1.5, 1, rng)[0]
                W_star_seq.append((t, seg_end - 1, 128))
            else:  # stationary
                for i in range(t, seg_end):
                    y[i] = (y[i-1] if i > 0 else 0.0) + rng.normal(0.0, 0.8)
                W_star_seq.append((t, seg_end - 1, 512))
            segments.append((t, seg_end - 1, kind))
            t = seg_end
            if t >= T:
                break
    return y.astype(float), segments, W_star_seq


# -----------------------------
# Metrics and evaluation utilities (used in evaluation too)
# -----------------------------

def sliding_window_means(x: np.ndarray, Ls: List[int]):
    x_t = torch.tensor(x, dtype=torch.float32).view(1, 1, -1)
    means = {}
    for L in Ls:
        kernel = torch.ones(1, 1, L) / L
        if x_t.shape[-1] < L:
            means[L] = np.array([])
        else:
            s = F.conv1d(x_t, kernel, padding=0)
            means[L] = s.view(-1).detach().cpu().numpy()
    return means


def run_length_distribution(err: np.ndarray):
    runs = []
    cnt = 0
    for e in err:
        if e == 1:
            cnt += 1
        else:
            if cnt > 0:
                runs.append(cnt)
            cnt = 0
    if cnt > 0:
        runs.append(cnt)
    return np.array(runs if len(runs) > 0 else [0], dtype=int)


def total_variation(arr: np.ndarray) -> float:
    if len(arr) <= 1:
        return 0.0
    return float(np.sum(np.abs(np.diff(arr))))


# -----------------------------
# Core runner for methods on a single series
# -----------------------------

def run_methods_on_series(y: np.ndarray, alpha: float = 0.1, seed: int = 0,
                          rsc_windows=(128, 512, 2048)) -> Dict[str, Dict[str, np.ndarray]]:
    T = len(y)
    fore = SeasonalNaiveEWMA(season_lag=24, bias_lag=169)

    # Initialize SAMC2
    sam_cfg = SAMC2Config(alpha=alpha)
    sam_ctrl = MultiResController(sam_cfg)
    sam_agg = SleepingExpertsAggregator(sam_cfg, num_scales=len(sam_cfg.scales))
    sam_cs = DyadicCSLayer(alpha)

    # Baselines
    pid = PIDController(alpha)
    rscs = {f"rsc{W}": RSC(alpha, W) for W in rsc_windows}
    cusum = CUSUMResetACI(alpha)

    names = ["samc2", "pid", *list(rscs.keys()), "cusum"]
    res = {nm: {"err": np.zeros(T, dtype=int), "q": np.zeros(T, dtype=float)} for nm in names}

    yhat = 0.0
    scores_recent: List[float] = []

    for t in range(T):
        yhat = fore.step(t, y[t - 1] if t > 0 else y[0])
        if t == 0:
            # Bootstrap first q estimates
            props, qmax = sam_ctrl.update_after_observation(0, 0.0)
            cs_press = sam_cs.pressures_for_scales(t, sam_ctrl.scales.numpy())
            q_sam, _ = sam_agg.step(props, 0, qmax, cs_press)
            res["samc2"]["q"][t] = q_sam
            res["pid"]["q"][t] = pid.update(0, 0.0)
            for nm, r in rscs.items():
                r.update(0.0)
                res[nm]["q"][t] = r.propose()
            res["cusum"]["q"][t] = cusum.update(0, 0.0)
            continue
        score = abs(y[t] - yhat)
        # Evaluate errors from previous step intervals
        for nm in names:
            res[nm]["err"][t] = int(abs(y[t] - yhat) > res[nm]["q"][t - 1])
        # Update SAMC2
        props, qmax = sam_ctrl.update_after_observation(res["samc2"]["err"][t], score)
        sam_cs.update(t, res["samc2"]["err"][t], alpha)
        cs_press = sam_cs.pressures_for_scales(t, sam_ctrl.scales.numpy())
        scores_recent.append(score)
        if len(scores_recent) > 1024:
            scores_recent.pop(0)
        sam_agg.update_scale(scores_recent)
        q_sam, _ = sam_agg.step(props, res["samc2"]["err"][t], qmax, cs_press)
        res["samc2"]["q"][t] = q_sam
        # Baselines updates
        res["pid"]["q"][t] = pid.update(res["pid"]["err"][t], score)
        for nm, r in rscs.items():
            r.update(score)
            res[nm]["q"][t] = r.propose()
        res["cusum"]["q"][t] = cusum.update(res["cusum"]["err"][t], score)

    return res


# -----------------------------
# High-level train entrypoints used by main.py
# -----------------------------

def train_experiment_1(y: np.ndarray, out_dir: str, alpha: float, seeds: List[int], rsc_windows: Tuple[int, int, int]) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []
    for seed in seeds:
        res = run_methods_on_series(y, alpha=alpha, seed=seed, rsc_windows=rsc_windows)
        out_path = os.path.join(out_dir, f"exp1_seed{seed}.npz")
        # Save arrays
        np.savez_compressed(
            out_path,
            samc2_err=res["samc2"]["err"], samc2_q=res["samc2"]["q"],
            pid_err=res["pid"]["err"], pid_q=res["pid"]["q"],
            cusum_err=res["cusum"]["err"], cusum_q=res["cusum"]["q"],
            **{f"{k}_err": v["err"] for k, v in res.items() if k.startswith("rsc")},
            **{f"{k}_q": v["q"] for k, v in res.items() if k.startswith("rsc")},
            alpha=alpha,
            rsc_windows=np.array(rsc_windows, dtype=int),
            seed=seed
        )
        saved_paths.append(out_path)
    return saved_paths


def train_experiment_2(y: np.ndarray, out_dir: str, alpha: float, rsc_windows: Tuple[int, int, int]) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    res = run_methods_on_series(y, alpha=alpha, seed=0, rsc_windows=rsc_windows)
    out_path = os.path.join(out_dir, "exp2_electricity.npz")
    np.savez_compressed(
        out_path,
        samc2_err=res["samc2"]["err"], samc2_q=res["samc2"]["q"],
        pid_err=res["pid"]["err"], pid_q=res["pid"]["q"],
        cusum_err=res["cusum"]["err"], cusum_q=res["cusum"]["q"],
        **{f"{k}_err": v["err"] for k, v in res.items() if k.startswith("rsc")},
        **{f"{k}_q": v["q"] for k, v in res.items() if k.startswith("rsc")},
        alpha=alpha,
        rsc_windows=np.array(rsc_windows, dtype=int)
    )
    return [out_path]


# Ablation helpers used for Experiment 3

def run_samc2_variants_for_ablation(y: np.ndarray, alpha: float = 0.1, variant: str = "full"):
    T = len(y)
    fore = SeasonalNaiveEWMA(season_lag=24, bias_lag=169)
    cfg = SAMC2Config(alpha=alpha)
    ctrl = MultiResController(cfg)
    agg = SleepingExpertsAggregator(cfg, num_scales=len(cfg.scales))
    cs = DyadicCSLayer(alpha)

    # variant toggles
    use_cs = True
    use_efficiency = True
    use_barrier_beta = True
    uniform_avg = False

    if variant == "no_cs":
        use_cs = False
    elif variant == "uniform_avg":
        uniform_avg = True
    elif variant == "no_efficiency":
        use_efficiency = False
    elif variant == "no_barrier":
        use_barrier_beta = False

    errs = np.zeros(T, dtype=int)
    qs = np.zeros(T, dtype=float)
    weights = np.zeros((T, len(cfg.scales)), dtype=float)
    per_expert_loss = np.zeros((T, len(cfg.scales)), dtype=float)

    yhat = 0.0
    scores_recent: List[float] = []

    for t in range(T):
        yhat = fore.step(t, y[t - 1] if t > 0 else y[0])
        if t == 0:
            props, qmax = ctrl.update_after_observation(0, 0.0)
            cs_press = cs.pressures_for_scales(t, ctrl.scales.numpy()) if use_cs else torch.zeros_like(props)
            if uniform_avg:
                w = torch.full_like(props, 1.0 / len(props))
                q = float(torch.dot(w, props))
                agg.w = w
                last_losses = torch.zeros_like(props)
            else:
                if not use_efficiency:
                    old_lam, old_beta = agg.cfg.lam, agg.cfg.barrier_beta
                    agg.cfg.lam = 1.0
                    agg.cfg.barrier_beta = 0.0 if not use_barrier_beta else old_beta
                    q, last_losses = agg.step(props, 0, qmax, cs_press if use_cs else None)
                    agg.cfg.lam = old_lam
                    agg.cfg.barrier_beta = old_beta
                else:
                    if not use_barrier_beta:
                        old_beta = agg.cfg.barrier_beta; agg.cfg.barrier_beta = 0.0
                        q, last_losses = agg.step(props, 0, qmax, cs_press if use_cs else None)
                        agg.cfg.barrier_beta = old_beta
                    else:
                        q, last_losses = agg.step(props, 0, qmax, cs_press if use_cs else None)
            qs[t] = q
            weights[t, :] = agg.w.detach().cpu().numpy()
            per_expert_loss[t, :] = last_losses.detach().cpu().numpy() if not uniform_avg else np.zeros(len(cfg.scales))
            continue

        score = abs(y[t] - yhat)
        err = int(abs(y[t] - yhat) > qs[t - 1])
        errs[t] = err
        props, qmax = ctrl.update_after_observation(err, score)
        if use_cs:
            cs.update(t, err, alpha)
            cs_press = cs.pressures_for_scales(t, ctrl.scales.numpy())
        else:
            cs_press = torch.zeros_like(props)
        scores_recent.append(score)
        if len(scores_recent) > 1024:
            scores_recent.pop(0)
        agg.update_scale(scores_recent)

        if uniform_avg:
            w = torch.full_like(props, 1.0 / len(props))
            q = float(torch.dot(w, props))
            last_losses = torch.zeros_like(props)
            agg.w = w
        else:
            if not use_efficiency:
                old_lam, old_beta = agg.cfg.lam, agg.cfg.barrier_beta
                agg.cfg.lam = 1.0
                agg.cfg.barrier_beta = 0.0 if not use_barrier_beta else old_beta
                q, last_losses = agg.step(props, err, qmax, cs_press if use_cs else None)
                agg.cfg.lam = old_lam
                agg.cfg.barrier_beta = old_beta
            else:
                if not use_barrier_beta:
                    old_beta = agg.cfg.barrier_beta; agg.cfg.barrier_beta = 0.0
                    q, last_losses = agg.step(props, err, qmax, cs_press if use_cs else None)
                    agg.cfg.barrier_beta = old_beta
                else:
                    q, last_losses = agg.step(props, err, qmax, cs_press if use_cs else None)

        qs[t] = q
        weights[t, :] = agg.w.detach().cpu().numpy()
        per_expert_loss[t, :] = last_losses.detach().cpu().numpy()

    return errs, qs, weights, per_expert_loss, list(SAMC2Config().scales)


def train_experiment_3(y: np.ndarray, out_dir: str, alpha: float) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    variants = ["full", "no_cs", "uniform_avg", "no_efficiency", "no_barrier"]
    paths = []
    for v in variants:
        errs, qs, weights, per_expert_loss, scales = run_samc2_variants_for_ablation(y, alpha=alpha, variant=v)
        out_path = os.path.join(out_dir, f"exp3_ablation_{v}.npz")
        np.savez_compressed(
            out_path,
            err=errs, q=qs, w=weights, loss_s=per_expert_loss, scales=np.array(scales, dtype=float), alpha=alpha, variant=v
        )
        paths.append(out_path)
    return paths
