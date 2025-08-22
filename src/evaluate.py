import os
import math
import time
import json
import yaml
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Matplotlib PDF font embedding (vector, suitable for academic papers)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class QueryCounter:
    def __init__(self, fn):
        self.fn = fn
        self.count = 0
    def __call__(self, S_batch: List[Set[int]]) -> np.ndarray:
        self.count += len(S_batch)
        return self.fn(S_batch)


# ------------------------ Synthetic pseudo-Boolean game ------------------------
class SyntheticGame:
    def __init__(self, d: int, beta_heavy: Dict[Tuple[int, ...], float], sigma_tail: float = 0.0, seed: int = 42):
        self.d = d
        self.beta_heavy = {tuple(sorted(U)): float(b) for U, b in beta_heavy.items()}
        self.sigma_tail = float(sigma_tail)
        self.rng = np.random.default_rng(seed)
    def _chi(self, r: np.ndarray, U: Tuple[int, ...]) -> np.ndarray:
        if len(U) == 0:
            return np.ones(r.shape[0], dtype=np.float64)
        return np.prod(r[:, list(U)], axis=1)
    def g_from_r(self, r_batch: np.ndarray) -> np.ndarray:
        B = r_batch.shape[0]
        vals = np.zeros(B, dtype=np.float64)
        for U, b in self.beta_heavy.items():
            vals += b * self._chi(r_batch, U)
        if self.sigma_tail > 0:
            vals += self.rng.normal(0.0, self.sigma_tail, size=B)
        return vals
    def v(self, S_batch: List[Set[int]]) -> np.ndarray:
        B = len(S_batch)
        r = -np.ones((B, self.d), dtype=np.int8)
        for i, S in enumerate(S_batch):
            if S:
                idx = list(S)
                r[i, idx] = 1
        return self.g_from_r(r)


# ------------------------ Transforms: Walsh -> Möbius -> Shapley ------------------------
def beta_to_m(beta: Dict[Tuple[int, ...], float], order_cap: Optional[int] = None) -> Dict[Tuple[int, ...], float]:
    m: Dict[Tuple[int, ...], float] = {}
    support = list(beta.keys())
    T_cands: List[Tuple[int, ...]] = []
    for U in support:
        if (order_cap is None) or (len(U) <= order_cap):
            T_cands.append(U)
    for T in T_cands:
        Tset = set(T)
        sgn_sum = 0.0
        for U, b in beta.items():
            Uset = set(U)
            if Uset.issubset(Tset):
                sgn_sum += ((-1.0) ** (len(Tset) - len(Uset))) * b
        m[T] = (2.0 ** len(T)) * sgn_sum
    return m

def m_to_phi(m: Dict[Tuple[int, ...], float], d: int) -> np.ndarray:
    phi = np.zeros(d, dtype=np.float64)
    for T, val in m.items():
        if len(T) == 0:
            continue
        w = val / float(len(T))
        for i in T:
            phi[i] += w
    return phi


# ------------------------ CS-SHAP core ------------------------
@dataclass
class CSShapConfig:
    max_queries: int
    order_trunc_k: int = 3
    mom_groups: int = 5
    order_skew: bool = True
    topL_singleton_factor: float = 5
    top_pairs_limit: Optional[int] = None
    top_triples_limit: Optional[int] = None
    tail_probes: int = 64
    seed: int = 42


class CSShap:
    def __init__(self, d: int, query_fn, cfg: CSShapConfig):
        self.d = d
        self.query_fn = query_fn
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.beta_hat: Dict[Tuple[int, ...], float] = {}
        self.beta_var: Dict[Tuple[int, ...], float] = {}
        self.m_hat: Dict[Tuple[int, ...], float] = {}
        self.m_var: Dict[Tuple[int, ...], float] = {}
        self.phi_hat: np.ndarray = np.zeros(d, dtype=np.float64)
        self.phi_ci: Optional[np.ndarray] = None
        self.tail_energy: float = 0.0
        self.queries_used: int = 0

    def _make_random_parities(self, M: int) -> np.ndarray:
        R = self.rng.integers(0, 2, size=(M, self.d), dtype=np.int8)
        R = 2 * R - 1
        return R

    def _parities_to_subsets(self, R: np.ndarray) -> List[Set[int]]:
        return [set(np.where(r == 1)[0].tolist()) for r in R]

    def _estimate_beta_singletons(self, R: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M = R.shape[0]
        Rt = torch.from_numpy(R.astype(np.float32))
        yt = torch.from_numpy(y.astype(np.float32)).view(M, 1)
        prods = Rt * yt
        g = max(1, self.cfg.mom_groups)
        m = M // g
        if m > 0:
            prods_trim = prods[:m*g]
            prods_groups = prods_trim.view(g, m, self.d)
            means = prods_groups.mean(dim=1)
            mom_vals, _ = torch.sort(means, dim=0)
            median_idx = g // 2
            beta_hat = mom_vals[median_idx, :].cpu().numpy()
            var = torch.median((means - mom_vals[median_idx:median_idx+1, :]) ** 2, dim=0).values.cpu().numpy() + 1e-12
        else:
            beta_hat = (prods.mean(dim=0)).cpu().numpy()
            var = (prods.var(dim=0)).cpu().numpy()
        return beta_hat.astype(np.float64), var.astype(np.float64)

    def _estimate_beta_pairs(self, R: np.ndarray, y: np.ndarray, idx_list: List[int], cap: Optional[int]) -> Tuple[Dict[Tuple[int, ...], float], Dict[Tuple[int, ...], float]]:
        M = R.shape[0]
        yt = torch.from_numpy(y.astype(np.float32)).view(M, 1)
        beta: Dict[Tuple[int, ...], float] = {}
        var: Dict[Tuple[int, ...], float] = {}
        g = max(1, self.cfg.mom_groups)
        m = M // g
        for ii in range(len(idx_list)):
            i = idx_list[ii]
            ri = torch.from_numpy(R[:, i].astype(np.float32)).view(M, 1)
            for jj in range(ii+1, len(idx_list)):
                j = idx_list[jj]
                rj = torch.from_numpy(R[:, j].astype(np.float32)).view(M, 1)
                prod = (ri * rj * yt).view(M)
                if m > 0:
                    prod_trim = prod[:m*g]
                    groups = prod_trim.view(g, m)
                    means = groups.mean(dim=1)
                    med = torch.median(means).item()
                    v = torch.median((means - med) ** 2).item() + 1e-12
                else:
                    med = float(prod.mean().item())
                    v = float(prod.var().item())
                U = (int(i), int(j))
                beta[U] = float(med)
                var[U] = float(v)
        if cap is not None and len(beta) > cap:
            order = sorted(beta.items(), key=lambda kv: -abs(kv[1]))
            keep = set(k for (k, _) in order[:cap])
            beta = {k: v for k, v in beta.items() if k in keep}
            var = {k: var[k] for k in keep}
        return beta, var

    def _estimate_beta_triples(self, R: np.ndarray, y: np.ndarray, idx_list: List[int], cap: Optional[int]) -> Tuple[Dict[Tuple[int, ...], float], Dict[Tuple[int, ...], float]]:
        M = R.shape[0]
        yt = torch.from_numpy(y.astype(np.float32)).view(M, 1)
        beta: Dict[Tuple[int, ...], float] = {}
        var: Dict[Tuple[int, ...], float] = {}
        g = max(1, self.cfg.mom_groups)
        m = M // g
        total = 0
        max_to_eval = cap if cap is not None else (len(idx_list) ** 3)
        for ii in range(len(idx_list)):
            i = idx_list[ii]
            ri = torch.from_numpy(R[:, i].astype(np.float32)).view(M, 1)
            for jj in range(ii+1, len(idx_list)):
                j = idx_list[jj]
                rj = torch.from_numpy(R[:, j].astype(np.float32)).view(M, 1)
                for kk in range(jj+1, len(idx_list)):
                    if total >= max_to_eval:
                        break
                    k = idx_list[kk]
                    rk = torch.from_numpy(R[:, k].astype(np.float32)).view(M, 1)
                    prod = (ri * rj * rk * yt).view(M)
                    if m > 0:
                        prod_trim = prod[:m*g]
                        groups = prod_trim.view(g, m)
                        means = groups.mean(dim=1)
                        med = torch.median(means).item()
                        v = torch.median((means - med) ** 2).item() + 1e-12
                    else:
                        med = float(prod.mean().item())
                        v = float(prod.var().item())
                    U = (int(i), int(j), int(k))
                    beta[U] = float(med)
                    var[U] = float(v)
                    total += 1
                if total >= max_to_eval:
                    break
            if total >= max_to_eval:
                break
        if cap is not None and len(beta) > cap:
            order = sorted(beta.items(), key=lambda kv: -abs(kv[1]))
            keep = set(k for (k, _) in order[:cap])
            beta = {k: v for k, v in beta.items() if k in keep}
            var = {k: var[k] for k in keep}
        return beta, var

    def _tail_energy_estimate(self, beta_hat: Dict[Tuple[int, ...], float]) -> float:
        M = self.cfg.tail_probes
        R = self._make_random_parities(M)
        S = self._parities_to_subsets(R)
        y = self.query_fn(S)
        self.queries_used += len(S)
        y_pred = np.zeros_like(y)
        for U, b in beta_hat.items():
            if len(U) == 0:
                y_pred += b
            else:
                chi = np.prod(R[:, list(U)], axis=1)
                y_pred += b * chi
        resid = y - y_pred
        return float(np.sqrt(np.mean(resid ** 2)))

    def run(self) -> Dict:
        start = time.time()
        M = max(16, int(0.8 * self.cfg.max_queries))
        R = self._make_random_parities(M)
        S = self._parities_to_subsets(R)
        y = self.query_fn(S)
        self.queries_used += len(S)
        beta1, var1 = self._estimate_beta_singletons(R, y)
        L1 = min(self.d, max(1, int(self.cfg.topL_singleton_factor * math.sqrt(self.cfg.max_queries))))
        if self.cfg.order_skew:
            idx_sorted = np.argsort(-np.abs(beta1))
        else:
            idx_sorted = np.arange(self.d)
            self.rng.shuffle(idx_sorted)
        top_idx = idx_sorted[:L1]
        for i in top_idx:
            U = (int(i),)
            self.beta_hat[U] = float(beta1[i])
            self.beta_var[U] = float(var1[i])
        if self.cfg.order_trunc_k >= 2:
            cap_pairs = self.cfg.top_pairs_limit
            if cap_pairs is None:
                cap_pairs = L1 * (L1 - 1) // 2
            beta2, var2 = self._estimate_beta_pairs(R, y, list(map(int, top_idx.tolist())), cap_pairs)
            for U, b in beta2.items():
                self.beta_hat[U] = float(b)
                self.beta_var[U] = float(var2[U])
        if self.cfg.order_trunc_k >= 3:
            L3 = min(L1, max(2, int(0.5 * L1)))
            idx3 = list(map(int, top_idx[:L3].tolist()))
            cap_triples = self.cfg.top_triples_limit
            if cap_triples is None:
                cap_triples = int(min(3 * L3, (L3 * (L3 - 1) * (L3 - 2)) // 6))
            beta3, var3 = self._estimate_beta_triples(R, y, idx3, cap_triples)
            for U, b in beta3.items():
                self.beta_hat[U] = float(b)
                self.beta_var[U] = float(var3[U])
        self.m_hat = beta_to_m(self.beta_hat, order_cap=self.cfg.order_trunc_k)
        self.m_var = {}
        for T in self.m_hat.keys():
            Tset = set(T)
            acc = 0.0
            for U, v in self.beta_var.items():
                if set(U).issubset(Tset):
                    acc += (2.0 ** len(T)) * math.sqrt(max(v, 0.0))
            self.m_var[T] = float(acc ** 2)
        self.phi_hat = m_to_phi(self.m_hat, self.d)
        self.tail_energy = self._tail_energy_estimate(self.beta_hat)
        phi_var = np.zeros(self.d, dtype=np.float64)
        for T, mv in self.m_var.items():
            if len(T) == 0:
                continue
            for i in T:
                phi_var[i] += mv / float((len(T) ** 2))
        contrib_counts = np.zeros(self.d)
        for T in self.m_hat.keys():
            for i in T:
                contrib_counts[i] += 1.0
        phi_var += (contrib_counts + 1.0) * (self.tail_energy ** 2)
        phi_std = np.sqrt(np.maximum(phi_var, 1e-12))
        z = 1.96
        self.phi_ci = np.vstack([self.phi_hat - z * phi_std, self.phi_hat + z * phi_std]).T
        end = time.time()
        print(f"[CS-SHAP] d={self.d}, queries_used={self.queries_used}, recovered_terms={len(self.beta_hat)}, tail_energy={self.tail_energy:.4g}, time={end-start:.2f}s")
        return {
            'beta_hat': self.beta_hat,
            'beta_var': self.beta_var,
            'm_hat': self.m_hat,
            'm_var': self.m_var,
            'phi_hat': self.phi_hat,
            'phi_ci': self.phi_ci,
            'tail_energy': self.tail_energy,
            'queries': self.queries_used,
        }


# ------------------------ Optional baselines ------------------------
def run_kernel_shap(query_fn_counted: QueryCounter, d: int, budget: int) -> Optional[np.ndarray]:
    try:
        import shap
    except Exception:
        print("[KernelSHAP] shap not available; skipping baseline.")
        return None
    M = max(10, int(0.9 * budget))
    def f_masks(M_mask):
        S_batch = [set(np.where(row == 1)[0].tolist()) for row in M_mask]
        vals = query_fn_counted(S_batch)
        return vals
    X_dummy = np.zeros((1, d))
    explainer = shap.KernelExplainer(f_masks, data=X_dummy, link='identity')
    x_explain = np.ones((1, d))
    try:
        shap_vals = explainer.shap_values(x_explain, nsamples=M)
        phi = np.array(shap_vals)[0]
        return phi
    except Exception as e:
        print("[KernelSHAP] Error during explanation:", e)
        return None


# ------------------------ Experiments from the paper draft ------------------------
def sample_heavy_terms(d: int, s: int, R: float, rng: np.random.Generator, order_dist=(0.6, 0.3, 0.1)) -> Dict[Tuple[int, ...], float]:
    sizes = rng.choice([1, 2, 3], size=s, p=list(order_dist))
    support = []
    used = set()
    for k in sizes:
        while True:
            U = tuple(sorted(rng.choice(d, size=k, replace=False).tolist()))
            if U not in used:
                used.add(U)
                support.append(U)
                break
    mags = np.exp(rng.uniform(np.log(1.0), np.log(R), size=s))
    signs = rng.choice([-1.0, 1.0], size=s)
    beta = {U: float(mag * sgn) for U, mag, sgn in zip(support, mags, signs)}
    return beta

def compute_ground_truth_phi_from_beta(beta: Dict[Tuple[int, ...], float], d: int) -> Tuple[Dict[Tuple[int, ...], float], np.ndarray]:
    m_true = beta_to_m(beta)
    phi_true = m_to_phi(m_true, d)
    return m_true, phi_true

def exp1_single_run(d=256, s=10, R=10.0, sigma=0.0, budgets: List[int] = None, seed=42, save_plots=True, outdir=".") -> Dict:
    print(f"[Exp1] Synthetic run: d={d}, s={s}, R={R}, sigma={sigma}")
    rng = np.random.default_rng(seed)
    beta = sample_heavy_terms(d, s, R, rng)
    game = SyntheticGame(d, beta, sigma_tail=sigma, seed=seed)
    m_true, phi_true = compute_ground_truth_phi_from_beta(beta, d)
    if budgets is None:
        base = int(12 * s * (int(math.log2(max(d, 2))) ** 2))
        budgets = [max(64, base // 2), base, base * 2]
    print("[Exp1] Budgets:", budgets)

    results = {"css": [], "kernel": []}
    errors_css, errors_kernel, q_css, q_kernel = [], [], [], []

    for b in budgets:
        cfg = CSShapConfig(max_queries=b, order_trunc_k=3, mom_groups=5, order_skew=True,
                           topL_singleton_factor=5.0, top_pairs_limit=5*s, top_triples_limit=3*s,
                           tail_probes=64, seed=seed)
        qc = QueryCounter(game.v)
        css = CSShap(d, qc, cfg)
        out = css.run()
        phi_hat = out['phi_hat']
        ci = out['phi_ci']
        l1_css = float(np.mean(np.abs(phi_hat - phi_true)))
        linf_css = float(np.max(np.abs(phi_hat - phi_true)))
        res_css = {"budget": b, "queries": qc.count, "l1": l1_css, "linf": linf_css}
        print(f"[Exp1][CS-SHAP] budget={b}, queries={qc.count}, L1={l1_css:.4g}, Linf={linf_css:.4g}")
        results["css"].append({**res_css, "phi": phi_hat, "ci": ci})
        errors_css.append(l1_css)
        q_css.append(qc.count)

        qc_k = QueryCounter(game.v)
        phi_kernel = run_kernel_shap(qc_k, d, qc.count)
        if phi_kernel is not None:
            l1_k = float(np.mean(np.abs(phi_kernel - phi_true)))
            linf_k = float(np.max(np.abs(phi_kernel - phi_true)))
            print(f"[Exp1][KernelSHAP] queries={qc_k.count}, L1={l1_k:.4g}, Linf={linf_k:.4g}")
            results["kernel"].append({"budget": b, "queries": qc_k.count, "l1": l1_k, "linf": linf_k, "phi": phi_kernel})
            errors_kernel.append(l1_k)
            q_kernel.append(qc_k.count)
        else:
            errors_kernel.append(np.nan)
            q_kernel.append(0)

    if save_plots:
        plt.figure(figsize=(5, 4))
        plt.plot(q_css, errors_css, marker='o', label='CS-SHAP')
        if not all(np.isnan(errors_kernel)) and any(np.isfinite(errors_kernel)):
            plt.plot(q_kernel, errors_kernel, marker='s', label='KernelSHAP')
        plt.xlabel('Model queries')
        plt.ylabel('L1 error on Shapley φ')
        plt.title('Error vs Queries (Synthetic)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(outdir, 'phi_error_css_vs_kernel.pdf')
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"[Exp1] Saved {fname}")

        phi_hat_last = results["css"][-1]["phi"]
        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(phi_true, phi_hat_last, s=10, alpha=0.7)
        lim = float(np.max(np.abs(np.concatenate([phi_true, phi_hat_last]))))
        lim = max(lim, 1e-6)
        plt.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
        plt.xlabel('True φ')
        plt.ylabel('CS-SHAP φ̂')
        plt.title('Shapley Scatter (last budget)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(outdir, 'phi_scatter_css.pdf')
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"[Exp1] Saved {fname}")

    return {"phi_true": phi_true, "results": results}


# ------------------------ Experiment 3: Interaction discovery ------------------------
def generate_quadratic_ground_truth(d=512, s_main=20, s_pair=15, R=50.0, seed=42):
    rng = np.random.default_rng(seed)
    idx_main = rng.choice(d, size=s_main, replace=False)
    pairs = set()
    while len(pairs) < s_pair:
        i, j = sorted(rng.choice(d, size=2, replace=False).tolist())
        if i != j:
            pairs.add((i, j))
    a = np.zeros(d)
    a[idx_main] = rng.choice([-1, 1], size=s_main) * np.exp(rng.uniform(np.log(1.0), np.log(R), size=s_main))
    b = {}
    for (i, j) in pairs:
        b[(i, j)] = float(rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(1.0), np.log(R))))
    a0 = 0.0
    return a0, a, b

def quadratic_value_function_factory(a0: float, a: np.ndarray, b: Dict[Tuple[int, int], float], x: np.ndarray):
    d = len(x)
    def v(S_batch: List[Set[int]]) -> np.ndarray:
        out = np.zeros(len(S_batch), dtype=np.float64)
        for t, S in enumerate(S_batch):
            val = a0
            if S:
                idx = list(S)
                val += float(np.sum(a[idx] * x[idx]))
                for (i, j), c in b.items():
                    if (i in S) and (j in S):
                        val += c * x[i] * x[j]
            out[t] = val
        return out
    return v

def ground_truth_m_phi_quadratic(x: np.ndarray, a0: float, a: np.ndarray, b: Dict[Tuple[int, int], float]) -> Tuple[Dict[Tuple[int, ...], float], np.ndarray]:
    beta = {}
    d = len(x)
    for i in range(d):
        if abs(a[i]) > 0:
            beta[(i,)] = float(a[i] * x[i])
    for (i, j), c in b.items():
        beta[(i, j)] = float(c * x[i] * x[j])
    m_true = beta_to_m(beta, order_cap=2)
    phi_true = m_to_phi(m_true, d)
    return m_true, phi_true

def exp3_single_run(d=512, s_main=10, s_pair=10, R=50.0, seed=7, budgets: List[int] = None, save_plots=True, outdir=".") -> Dict:
    print(f"[Exp3] Interaction discovery: d={d}, s_main={s_main}, s_pair={s_pair}, R={R}")
    rng = np.random.default_rng(seed)
    a0, a, b = generate_quadratic_ground_truth(d, s_main, s_pair, R, seed)
    x = rng.normal(0, 1, size=d).astype(np.float64)
    v = quadratic_value_function_factory(a0, a, b, x)
    m_true, phi_true = ground_truth_m_phi_quadratic(x, a0, a, b)

    if budgets is None:
        base = int(12 * (s_main + s_pair) * (int(math.log2(max(d, 2))) ** 2))
        budgets = [max(64, base // 2), base]
    print("[Exp3] Budgets:", budgets)

    css_res = []
    for bgt in budgets:
        cfg = CSShapConfig(max_queries=bgt, order_trunc_k=2, mom_groups=5, order_skew=True,
                           topL_singleton_factor=5.0, top_pairs_limit=5*(s_main+s_pair), top_triples_limit=0,
                           tail_probes=64, seed=seed)
        qc = QueryCounter(v)
        css = CSShap(d, qc, cfg)
        out = css.run()
        phi_hat = out['phi_hat']
        l1 = float(np.mean(np.abs(phi_hat - phi_true)))
        css_res.append({"queries": qc.count, "l1": l1, "phi": phi_hat, "beta_hat": out['beta_hat']})
        print(f"[Exp3][Order-Skew] queries={qc.count}, L1={l1:.4g}")

    uni_res = []
    for bgt in budgets:
        cfg = CSShapConfig(max_queries=bgt, order_trunc_k=2, mom_groups=5, order_skew=False,
                           topL_singleton_factor=5.0, top_pairs_limit=5*(s_main+s_pair), top_triples_limit=0,
                           tail_probes=64, seed=seed)
        qc = QueryCounter(v)
        css = CSShap(d, qc, cfg)
        out = css.run()
        phi_hat = out['phi_hat']
        l1 = float(np.mean(np.abs(phi_hat - phi_true)))
        uni_res.append({"queries": qc.count, "l1": l1, "phi": phi_hat, "beta_hat": out['beta_hat']})
        print(f"[Exp3][Uniform] queries={qc.count}, L1={l1:.4g}")

    true_pairs = set(b.keys())
    def rank_pairs(beta_hat: Dict[Tuple[int, ...], float]) -> List[Tuple[Tuple[int, int], float]]:
        pairs = [(U, abs(val)) for U, val in beta_hat.items() if len(U) == 2]
        pairs.sort(key=lambda kv: -kv[1])
        return pairs

    pairs_skew = rank_pairs(css_res[-1]["beta_hat"])  # last budget
    pairs_uni = rank_pairs(uni_res[-1]["beta_hat"])   # last budget

    def precision_recall_at_k(ranked: List[Tuple[Tuple[int, int], float]], true_set: Set[Tuple[int, int]], K: int) -> Tuple[float, float]:
        sel = [p for p, _ in ranked[:K]]
        tp = len([p for p in sel if p in true_set])
        prec = tp / max(K, 1)
        rec = tp / max(len(true_set), 1)
        return prec, rec

    Ks = list(range(1, min(20, max(2, len(true_pairs))) + 1))
    prec_skew, rec_skew = [], []
    prec_uni, rec_uni = [], []
    for K in Ks:
        p, r = precision_recall_at_k(pairs_skew, true_pairs, K)
        prec_skew.append(p); rec_skew.append(r)
        p2, r2 = precision_recall_at_k(pairs_uni, true_pairs, K)
        prec_uni.append(p2); rec_uni.append(r2)

    if save_plots:
        plt.figure(figsize=(5, 4))
        plt.plot(rec_skew, prec_skew, marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Interaction PR (Order-Skewed)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname1 = os.path.join(outdir, 'interaction_pr_order_skew_pair1.pdf')
        plt.savefig(fname1, bbox_inches='tight')
        plt.close()
        print(f"[Exp3] Saved {fname1}")

        plt.figure(figsize=(5, 4))
        plt.plot(rec_uni, prec_uni, marker='s', color='tab:orange')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Interaction PR (Uniform Nomination)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname2 = os.path.join(outdir, 'interaction_pr_uniform_pair2.pdf')
        plt.savefig(fname2, bbox_inches='tight')
        plt.close()
        print(f"[Exp3] Saved {fname2}")

    return {
        "phi_true": phi_true,
        "css": css_res,
        "uniform": uni_res,
        "true_pairs": list(true_pairs),
        "PR_skew": {"K": Ks, "precision": prec_skew, "recall": rec_skew},
        "PR_uniform": {"K": Ks, "precision": prec_uni, "recall": rec_uni},
    }


# ------------------------ Model-based evaluation (trained MLP) ------------------------
class MLPWrapper:
    def __init__(self, model_path: str):
        ckpt = torch.load(model_path, map_location="cpu")
        self.d_in = int(ckpt["d_in"]) if "d_in" in ckpt else None
        hidden_dims = tuple(ckpt.get("hidden_dims", [128, 64]))
        from train import MLPRegressor  # reuse architecture
        self.model = MLPRegressor(self.d_in, hidden_dims)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.x_mean = ckpt["x_mean"].astype(np.float32)
        self.x_std = ckpt["x_std"].astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = (X.astype(np.float32) - self.x_mean) / (self.x_std + 1e-6)
        with torch.no_grad():
            y = self.model(torch.from_numpy(Xn)).cpu().numpy().astype(np.float64)
        return y


def masking_query_fn(model: MLPWrapper, x0: np.ndarray, xbar: np.ndarray):
    d = x0.shape[0]
    def v(S_batch: List[Set[int]]) -> np.ndarray:
        B = len(S_batch)
        X = np.tile(xbar.reshape(1, -1), (B, 1))
        for b, S in enumerate(S_batch):
            if len(S) > 0:
                idx = list(S)
                X[b, idx] = x0[idx]
        y = model.predict(X)
        return y
    return v


def evaluate_trained_model(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg.get("data", {}).get("dir", "data")
    models_dir = cfg.get("model", {}).get("dir", "models")
    images_dir = cfg.get("experiment", {}).get("images_dir", ".research/iteration1/images")
    os.makedirs(images_dir, exist_ok=True)

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    data_name = cfg.get("data", {}).get("name", "synthetic_tabular")
    data_path = os.path.join(data_dir, f"{data_name}.npz")
    npz = np.load(data_path)
    X_test = npz["X_test"].astype(np.float64)
    y_test = npz["y_test"].astype(np.float64)

    model_path = os.path.join(models_dir, "mlp_regressor.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train.py first.")
    model = MLPWrapper(model_path)

    # Choose an instance to explain
    x0 = X_test[0]
    xbar = np.zeros_like(x0)  # interventional baseline
    d = x0.shape[0]

    v_fn = masking_query_fn(model, x0, xbar)
    qc = QueryCounter(v_fn)

    budgets = cfg.get("csshap", {}).get("budgets", [200, 400])
    css_cfg_common = dict(
        order_trunc_k=int(cfg.get("csshap", {}).get("order_trunc_k", 3)),
        mom_groups=int(cfg.get("csshap", {}).get("mom_groups", 5)),
        order_skew=bool(cfg.get("csshap", {}).get("order_skew", True)),
        topL_singleton_factor=float(cfg.get("csshap", {}).get("topL_singleton_factor", 4.0)),
        tail_probes=int(cfg.get("csshap", {}).get("tail_probes", 64)),
        seed=seed,
    )

    for b in budgets:
        cfg_css = CSShapConfig(max_queries=int(b), **css_cfg_common)
        css = CSShap(d, qc, cfg_css)
        out = css.run()
        phi = out["phi_hat"]
        ci = out["phi_ci"]
        order = np.argsort(-np.abs(phi))
        topk = min(20, d)
        idx = order[:topk]

        # Bar plot with CI
        plt.figure(figsize=(6, 4))
        yvals = phi[idx]
        yerr = np.vstack([phi[idx] - ci[idx, 0], ci[idx, 1] - phi[idx]])
        plt.bar(range(topk), yvals, yerr=yerr, capsize=3)
        plt.xlabel("Feature (sorted by |φ|)")
        plt.ylabel("Shapley value φ")
        plt.title(f"CS-SHAP on MLP (budget={b}, queries={out['queries']})")
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(images_dir, f"mlp_csshap_bar_budget{b}.pdf")
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"[Eval-MLP] Saved {fname}")

        # Optional: compare vs KernelSHAP if available
        qc_k = QueryCounter(v_fn)
        phi_kernel = run_kernel_shap(qc_k, d, out['queries'])
        if phi_kernel is not None:
            plt.figure(figsize=(4.5, 4.5))
            lim = float(np.max(np.abs(np.concatenate([phi, phi_kernel]))))
            lim = max(lim, 1e-6)
            plt.scatter(phi_kernel, phi, s=10, alpha=0.7)
            plt.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
            plt.xlabel('KernelSHAP φ (baseline)')
            plt.ylabel('CS-SHAP φ̂')
            plt.title('MLP: CS-SHAP vs KernelSHAP')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = os.path.join(images_dir, f"mlp_csshap_vs_kernel_budget{b}.pdf")
            plt.savefig(fname, bbox_inches='tight')
            plt.close()
            print(f"[Eval-MLP] Saved {fname}")

    # Print a small summary
    print(f"[Eval-MLP] Completed CS-SHAP runs for budgets={budgets}. Total model queries used={qc.count} (across all runs, counted by wrapper).")


# ------------------------ Quick functionality tests (Exp1 & Exp3) ------------------------
def run_synthetic_experiments_from_config(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    images_dir = cfg.get("experiment", {}).get("images_dir", ".research/iteration1/images")
    os.makedirs(images_dir, exist_ok=True)
    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    exp1_cfg = cfg.get("experiments", {}).get("exp1", {})
    res1 = exp1_single_run(
        d=int(exp1_cfg.get("d", 64)),
        s=int(exp1_cfg.get("s", 5)),
        R=float(exp1_cfg.get("R", 10.0)),
        sigma=float(exp1_cfg.get("sigma", 0.01)),
        budgets=list(map(int, exp1_cfg.get("budgets", [128, 256]))),
        seed=int(exp1_cfg.get("seed", seed)),
        save_plots=True,
        outdir=images_dir,
    )
    print("[Evaluate] Exp1 completed. Example φ_true[:10]:", np.round(res1["phi_true"][:10], 4))

    exp3_cfg = cfg.get("experiments", {}).get("exp3", {})
    res3 = exp3_single_run(
        d=int(exp3_cfg.get("d", 128)),
        s_main=int(exp3_cfg.get("s_main", 8)),
        s_pair=int(exp3_cfg.get("s_pair", 8)),
        R=float(exp3_cfg.get("R", 30.0)),
        seed=int(exp3_cfg.get("seed", seed)),
        budgets=list(map(int, exp3_cfg.get("budgets", [200, 400]))),
        save_plots=True,
        outdir=images_dir,
    )
    print("[Evaluate] Exp3 completed. True pair count:", len(res3["true_pairs"]))


if __name__ == "__main__":
    # Run both synthetic experiments and trained-model evaluation
    run_synthetic_experiments_from_config()
    evaluate_trained_model()
