import os
import math
from typing import List, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.train import sliding_window_means, run_length_distribution, total_variation


def ensure_pdf_style():
    plt.rcParams.update({
        'pdf.fonttype': 42,  # TrueType
        'ps.fonttype': 42,
        'figure.dpi': 200,
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.constrained_layout.use': True,
    })
    sns.set_context('paper')
    sns.set_style('whitegrid')


def evaluate_exp1(model_paths: List[str], img_dir: str, alpha: float):
    ensure_pdf_style()
    os.makedirs(img_dir, exist_ok=True)
    Ls = [16, 32, 64, 128, 256, 512]
    methods = ["samc2", "pid", "cusum"]

    # Collect metrics across seeds
    metrics_accum: Dict[str, Dict[str, list]] = {}
    all_rsc = set()

    for p in model_paths:
        data = np.load(p, allow_pickle=True)
        # discover RSC keys
        for k in data.files:
            if k.endswith('_err') and k.startswith('rsc'):
                all_rsc.add(k[:-4])
    methods.extend(sorted(list(all_rsc)))

    metrics_accum = {m: {"avg_dev": [], "sup_dev": [], "frac90_L32": [], "mean_width": [],
                         "p_run_ge_5": [], "tv": [], "std_q": []} for m in methods}

    for p in model_paths:
        data = np.load(p, allow_pickle=True)
        # Per-method arrays
        res = {}
        for m in methods:
            err_key = f"{m}_err"
            q_key = f"{m}_q"
            if err_key in data.files and q_key in data.files:
                res[m] = {"err": data[err_key], "q": data[q_key]}
        # Metrics per method
        for m in res.keys():
            err = res[m]["err"]
            q = res[m]["q"]
            win_means = sliding_window_means(err.astype(float), Ls)
            avg_dev = {L: float(np.mean(np.abs(win_means[L] - alpha))) if win_means[L].size > 0 else 0.0 for L in Ls}
            sup_dev = {L: float(np.max(np.abs(win_means[L] - alpha))) if win_means[L].size > 0 else 0.0 for L in Ls}
            frac90_L32 = 0.0
            wm32 = sliding_window_means(err.astype(float), [32])[32]
            if wm32.size > 0:
                frac90_L32 = float(np.mean((wm32 >= alpha - 0.05) & (wm32 <= alpha + 0.05)))
            runs = run_length_distribution(err.astype(int))
            p_run_ge_5 = float(np.mean(runs >= 5)) if runs.size > 0 else 0.0
            mean_width = float(np.mean(q))
            tv = total_variation(q)
            std_q = float(np.std(q))

            metrics_accum[m]["avg_dev"].append(avg_dev[64])
            metrics_accum[m]["sup_dev"].append(sup_dev[64])
            metrics_accum[m]["frac90_L32"].append(frac90_L32)
            metrics_accum[m]["mean_width"].append(mean_width)
            metrics_accum[m]["p_run_ge_5"].append(p_run_ge_5)
            metrics_accum[m]["tv"].append(tv)
            metrics_accum[m]["std_q"].append(std_q)

        # Plots per seed
        plt.figure(figsize=(6, 4))
        label_map = {"samc2": "SAMC2", "pid": "PID/ACI", "cusum": "CUSUM-ACI"}
        for m in methods:
            if f"{m}_err" in data.files:
                err = data[f"{m}_err"].astype(float)
                means_L = [float(np.mean(np.abs(sliding_window_means(err, [L])[L] - alpha))) for L in Ls]
                plt.plot(Ls, means_L, marker='o', label=label_map.get(m, m.upper()))
        plt.xlabel("Window length L")
        plt.ylabel("Avg |coverage - alpha|")
        plt.title("Exp1 coverage deviation vs L")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        bn = os.path.splitext(os.path.basename(p))[0]
        plt.savefig(os.path.join(img_dir, f"window_coverage_deviation_{bn}.pdf"), bbox_inches="tight")
        plt.close()

        # Run length survival curve
        plt.figure(figsize=(6, 4))
        for m in methods:
            if f"{m}_err" in data.files:
                runs = run_length_distribution(data[f"{m}_err"].astype(int))
                if runs.size == 0:
                    continue
                xs = np.sort(np.unique(runs))
                surv = [np.mean(runs >= k) for k in xs]
                plt.step(xs, surv, where='post', label=m.upper())
        plt.xlabel("Run length k (consecutive misses)")
        plt.ylabel("P(run ≥ k)")
        plt.title("Exp1 miss streak survival")
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, f"run_length_survival_{bn}.pdf"), bbox_inches="tight")
        plt.close()

    # Print summary
    print("[Exp1] Summary across seeds (mean ± std) for L=64 representative:")
    for m in methods:
        if len(metrics_accum[m]["avg_dev"]) == 0:
            continue
        avg_dev_m = np.mean(metrics_accum[m]["avg_dev"]) ; avg_dev_s = np.std(metrics_accum[m]["avg_dev"])
        sup_dev_m = np.mean(metrics_accum[m]["sup_dev"]) ; sup_dev_s = np.std(metrics_accum[m]["sup_dev"])
        frac_m = np.mean(metrics_accum[m]["frac90_L32"]) ; frac_s = np.std(metrics_accum[m]["frac90_L32"]) 
        width_m = np.mean(metrics_accum[m]["mean_width"]) ; width_s = np.std(metrics_accum[m]["mean_width"]) 
        prun_m = np.mean(metrics_accum[m]["p_run_ge_5"]) ; prun_s = np.std(metrics_accum[m]["p_run_ge_5"]) 
        tv_m = np.mean(metrics_accum[m]["tv"]) ; tv_s = np.std(metrics_accum[m]["tv"]) 
        stdq_m = np.mean(metrics_accum[m]["std_q"]) ; stdq_s = np.std(metrics_accum[m]["std_q"]) 
        print(f"  {m:8s} | avg_dev64={avg_dev_m:.4f}±{avg_dev_s:.4f} | sup_dev64={sup_dev_m:.4f}±{sup_dev_s:.4f} | " +
              f"frac_in[α±0.05](L≈32)={frac_m:.3f}±{frac_s:.3f} | mean_q={width_m:.3f}±{width_s:.3f} | P(run≥5)={prun_m:.3f}±{prun_s:.3f} | TV(q)={tv_m:.1f}±{tv_s:.1f} | std(q)={stdq_m:.3f}±{stdq_s:.3f}")

    print(f"[Exp1] Plots saved in {img_dir}")


def evaluate_exp2(model_path: str, img_dir: str, alpha: float):
    ensure_pdf_style()
    os.makedirs(img_dir, exist_ok=True)
    data = np.load(model_path, allow_pickle=True)
    methods = ["samc2", "pid", "cusum"]
    for k in data.files:
        if k.endswith('_err') and k.startswith('rsc'):
            m = k[:-4]
            if m not in methods:
                methods.append(m)
    Ls = [24, 48, 72, 168]

    # Coverage deviation vs L
    plt.figure(figsize=(6, 4))
    for m in methods:
        err = data[f"{m}_err"].astype(float)
        means_L = [float(np.mean(np.abs(sliding_window_means(err, [L])[L] - alpha))) for L in Ls]
        plt.plot(Ls, means_L, marker='o', label=m.upper())
    plt.xlabel("Window length L (hours)")
    plt.ylabel("Avg |coverage - alpha|")
    plt.title("Exp2 coverage deviation vs L (electricity)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "window_coverage_deviation_electricity.pdf"), bbox_inches="tight")
    plt.close()

    # Streak survival
    plt.figure(figsize=(6, 4))
    for m in methods:
        runs = run_length_distribution(data[f"{m}_err"].astype(int))
        if runs.size == 0:
            continue
        xs = np.sort(np.unique(runs))
        surv = [np.mean(runs >= k) for k in xs]
        plt.step(xs, surv, where='post', label=m.upper())
    plt.xlabel("Run length k")
    plt.ylabel("P(run ≥ k)")
    plt.title("Exp2 miss streak survival (electricity)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "run_length_survival_electricity.pdf"), bbox_inches="tight")
    plt.close()

    # Print summary
    print("[Exp2] Summary metrics:")
    for m in methods:
        err = data[f"{m}_err"].astype(float)
        q = data[f"{m}_q"].astype(float)
        frac24 = 0.0; frac48=0.0; frac72=0.0
        wm24 = sliding_window_means(err, [24])[24]
        wm48 = sliding_window_means(err, [48])[48]
        wm72 = sliding_window_means(err, [72])[72]
        if wm24.size>0: frac24 = float(np.mean((wm24 >= alpha - 0.05) & (wm24 <= alpha + 0.05)))
        if wm48.size>0: frac48 = float(np.mean((wm48 >= alpha - 0.05) & (wm48 <= alpha + 0.05)))
        if wm72.size>0: frac72 = float(np.mean((wm72 >= alpha - 0.05) & (wm72 <= alpha + 0.05)))
        runs = run_length_distribution(err)
        prun5 = float(np.mean(runs >= 5)) if runs.size > 0 else 0.0
        mean_q = float(np.mean(q))
        std_q = float(np.std(q))
        tv_q = total_variation(q)
        print(f"  {m:8s} | frac_in[α±0.05]: L24={frac24:.3f}, L48={frac48:.3f}, L72={frac72:.3f} | P(run≥5)={prun5:.3f} | mean_q={mean_q:.3f} | std_q={std_q:.3f} | TV(q)={tv_q:.1f}")

    print(f"[Exp2] Plots saved in {img_dir}")


def evaluate_exp3(model_paths: List[str], img_dir: str):
    ensure_pdf_style()
    os.makedirs(img_dir, exist_ok=True)

    # Load variants
    results = {}
    for p in model_paths:
        d = np.load(p, allow_pickle=True)
        v = str(d['variant'])
        results[v] = {"err": d['err'], "q": d['q'], "w": d['w'], "loss_s": d['loss_s'], "scales": d['scales']}

    variants = list(results.keys())
    Ls = [32, 128, 512, 2048]
    reg_stats = {v: [] for v in variants}

    for v in variants:
        loss_s = results[v]["loss_s"]  # T x S
        if loss_s.ndim != 2:
            continue
        w = results[v]["w"]
        agg_loss = np.sum(w * loss_s, axis=1)
        agg_sums = np.cumsum(agg_loss)
        sums = np.cumsum(loss_s, axis=0)
        for L in Ls:
            if loss_s.shape[0] <= L:
                continue
            reg_list = []
            for t in range(L, loss_s.shape[0]):
                sum_win = sums[t, :] - sums[t - L, :]
                oracle = float(np.min(sum_win))
                agg_win = float(agg_sums[t] - agg_sums[t - L])
                reg = agg_win - oracle
                reg_list.append(reg / math.sqrt(L))
            if len(reg_list) > 0:
                reg_stats[v].append((L, float(np.median(reg_list))))

    # Plot normalized regret
    plt.figure(figsize=(6, 4))
    for v in variants:
        xs = [L for (L, med) in reg_stats[v]]
        ys = [med for (L, med) in reg_stats[v]]
        if len(xs) > 0:
            plt.plot(xs, ys, marker='o', label=v)
    plt.xlabel("Window length L")
    plt.ylabel("Median Reg_I/√L")
    plt.title("Exp3 strongly adaptive regret (median)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "regret_normalized_ablation.pdf"), bbox_inches="tight")
    plt.close()

    # Pivot time: compute after each 10k-segment using heuristic: choose target scale index as argmin avg per-expert loss over the segment
    # For demonstration, if true W* not available, we approximate.
    plt.figure(figsize=(6, 4))
    pivot_times = {}
    for v in variants:
        w = results[v]["w"]
        T = w.shape[0]
        seg_bounds = list(range(0, T, 10000)) + [T]
        pts = []
        for i in range(len(seg_bounds)-1):
            a, b = seg_bounds[i], seg_bounds[i+1]-1
            idx = int(np.argmax(np.mean(w[a:b+1, :], axis=0)))
            subw = w[a:b+1, idx]
            hit = np.where(subw >= 0.5)[0]
            pt = int(hit[0]) if hit.size > 0 else (b - a + 1)
            pts.append(pt)
        if len(pts)==0:
            pts=[0]
        pivot_times[v] = pts
    medians = [float(np.median(pivot_times[v])) for v in variants]
    sns.barplot(x=variants, y=medians, color='#4C72B0')
    plt.ylabel("Median pivot time (steps)")
    plt.title("Exp3 pivot time by variant")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "pivot_time_distribution_ablation.pdf"), bbox_inches="tight")
    plt.close()

    # Width stability: TV(q) and max |Δq|
    plt.figure(figsize=(6, 4))
    tvs = [float(np.sum(np.abs(np.diff(results[v]["q"])))) if len(results[v]["q"])>1 else 0.0 for v in variants]
    max_dq = [float(np.max(np.abs(np.diff(results[v]["q"])))) if len(results[v]["q"]) > 1 else 0.0 for v in variants]
    x = np.arange(len(variants))
    plt.bar(x - 0.2, tvs, width=0.4, label='TV(q)')
    plt.bar(x + 0.2, max_dq, width=0.4, label='max |Δq|')
    plt.xticks(x, variants)
    plt.ylabel("Stability metrics")
    plt.title("Exp3 width stability by variant")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "width_stability_ablation.pdf"), bbox_inches="tight")
    plt.close()

    print(f"[Exp3] Plots saved in {img_dir}")
