import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import auc

def plot_curves(results_df, optimizers_list, benchmark_name, time_limit, output_dir):
    bench_df = results_df[results_df['benchmark'] == benchmark_name].copy()
    if bench_df.empty:
        print(f"No data to plot for benchmark {benchmark_name}.")
        return

    # Performance Curve
    plt.figure(figsize=(10, 7))
    sns.lineplot(data=bench_df, x='wall_clock_time', y='best_found_value', hue='optimizer', errorbar=('ci', 95), estimator='mean', hue_order=optimizers_list)
    plt.xlabel('Wall-Clock Time (s)')
    plt.ylabel('Best Objective Value Found (log scale)')
    plt.title(f'HPO Performance on {benchmark_name}')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(title='Optimizer')
    plt.yscale('log')
    plt.xlim(0, time_limit)
    plt.tight_layout()
    perf_filename = os.path.join(output_dir, f"performance_curve_{benchmark_name.replace('-', '_')}.pdf")
    plt.savefig(perf_filename, bbox_inches="tight")
    print(f"Saved performance plot to {perf_filename}")
    plt.close()

    # Overhead Curve
    plt.figure(figsize=(10, 7))
    sns.lineplot(data=bench_df, x='wall_clock_time', y='scheduler_overhead', hue='optimizer', errorbar=('ci', 95), estimator='mean', hue_order=optimizers_list)
    plt.xlabel('Wall-Clock Time (s)')
    plt.ylabel('Cumulative Scheduler Overhead (s)')
    plt.title(f'Scheduler Overhead on {benchmark_name}')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(title='Optimizer')
    plt.xlim(0, time_limit)
    plt.tight_layout()
    overhead_filename = os.path.join(output_dir, f"overhead_curve_{benchmark_name.replace('-', '_')}.pdf")
    plt.savefig(overhead_filename, bbox_inches="tight")
    print(f"Saved overhead plot to {overhead_filename}")
    plt.close()

def analyze_results(results_df, optimizers_list, time_limit):
    print("\n--- Analysis: Area Under Curve (AUC) and Statistical Tests ---")
    auc_results = []
    time_grid = np.linspace(0, time_limit, 500)

    for group_keys, group_df in results_df.groupby(['optimizer', 'benchmark', 'seed']):
        if group_df.empty: continue
        f_fill = pd.Series(group_df['best_found_value'].values, index=group_df['wall_clock_time'].values).reindex(time_grid, method='ffill').fillna(method='bfill')
        log_f_fill = np.log(f_fill)
        auc_val = auc(time_grid, log_f_fill)
        auc_results.append({'optimizer': group_keys[0], 'benchmark': group_keys[1], 'seed': group_keys[2], 'auc': auc_val})

    if not auc_results:
        print("Could not compute AUC results.")
        return
        
    auc_df = pd.DataFrame(auc_results)
    
    for benchmark in auc_df['benchmark'].unique():
        print(f"\n--- Benchmark: {benchmark} ---")
        bench_auc = auc_df[auc_df['benchmark'] == benchmark]
        mean_aucs = bench_auc.groupby('optimizer')['auc'].mean().sort_values()
        print("Mean Log-AUC (lower is better):")
        print(mean_aucs)
        
        car_hpo_aucs = bench_auc[bench_auc['optimizer'] == 'CAR-HPO']['auc'].values
        if len(car_hpo_aucs) == 0:
            print("No results for CAR-HPO to compare against.")
            continue

        print("\nWilcoxon Signed-Rank Test (vs CAR-HPO):")
        for opt_name in optimizers_list:
            if opt_name == 'CAR-HPO': continue
            baseline_aucs = bench_auc[bench_auc['optimizer'] == opt_name]['auc'].values
            if len(baseline_aucs) == len(car_hpo_aucs) and len(baseline_aucs) > 0:
                try:
                    stat, p_value = wilcoxon(car_hpo_aucs, baseline_aucs, alternative='less')
                    print(f"  - vs {opt_name}: p-value = {p_value:.4f} (Is CAR-HPO's AUC significantly lower?)")
                except ValueError as e:
                     print(f"  - vs {opt_name}: Could not perform test. Reason: {e}")
            else:
                print(f"  - vs {opt_name}: Inconsistent number of seeds or no data, skipping test.")
