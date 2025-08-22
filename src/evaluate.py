import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import random

def benchmark_sampler(sampler, sampler_name, n_gen_samples, intervention_var, intervention_val, n_runs, d, t_train):
    """Benchmarks a single sampler's performance."""
    run_times = []
    try:
        for i in range(n_runs):
            print(f"    Running {sampler_name} (run {i+1}/{n_runs})...")
            start_sample_time = time.time()
            _ = sampler.sample(
                num_samples=n_gen_samples,
                intervention_var=intervention_var,
                intervention_val=intervention_val
            )
            run_times.append(time.time() - start_sample_time)
        
        mean_time = np.mean(run_times)
        std_time = np.std(run_times)
    except TimeoutError as e:
        print(f"    {sampler_name} failed: {e}")
        mean_time, std_time = np.nan, np.nan
    except Exception as e:
        print(f"    An error occurred with {sampler_name}: {e}")
        mean_time, std_time = np.nan, np.nan

    result = {
        'sampler': sampler_name,
        'd': d,
        'n_gen': n_gen_samples,
        't_train': t_train,
        't_sample_mean': mean_time,
        't_sample_std': std_time
    }
    print(f"    -> {sampler_name} mean time: {mean_time if not np.isnan(mean_time) else 'N/A'}")
    return result

def choose_intervention_variable(graph, data):
    """Selects a suitable intervention variable from the graph."""
    non_root_leaf = [n for n in graph.nodes() if graph.in_degree(n) > 0 and graph.out_degree(n) > 0]
    if not non_root_leaf:
        non_root_leaf = [n for n in graph.nodes() if graph.in_degree(n) > 0]
        if not non_root_leaf:
             non_root_leaf = list(graph.nodes())[1:]
    intervention_var = random.choice(non_root_leaf)
    intervention_val = data[intervention_var].mean() + data[intervention_var].std()
    return intervention_var, intervention_val

def plot_results(df, image_dir, plot_config):
    """Generates and saves plots based on the experiment results."""
    print("\nGenerating plots...")
    os.makedirs(image_dir, exist_ok=True)
    if df.empty:
        print("Results DataFrame is empty, skipping plotting.")
        return

    sns.set_theme(style="whitegrid", context="paper")

    # Plot 1: Time vs. Dimensionality
    plt.figure(figsize=(6, 4))
    fixed_n_samples = plot_config['fixed_n_samples_for_d_plot']
    plot_data = df[df['n_gen'] == fixed_n_samples].copy()
    plot_data.dropna(subset=['t_sample_mean'], inplace=True)

    if not plot_data.empty:
        ax = sns.lineplot(data=plot_data, x='d', y='t_sample_mean', hue='sampler', marker='o', errorbar='sd')
        ax.set_yscale('log')
        ax.set_xlabel("Number of Variables (d)")
        ax.set_ylabel("Mean Sampling Time (s, log scale)")
        ax.set_title(f"Scalability with Dimensionality (N_samples = {fixed_n_samples})")
        plt.legend(title='Sampler')
        plt.tight_layout()
        filename = os.path.join(image_dir, "sampling_time_vs_dimensionality.pdf")
        plt.savefig(filename, bbox_inches="tight")
        print(f"Saved plot: {filename}")
    else:
        print(f"No data for n_gen={fixed_n_samples}, skipping dimensionality plot.")
    plt.close()

    # Plot 2: Time vs. Number of Samples
    plt.figure(figsize=(6, 4))
    fixed_d = plot_config['fixed_d_for_n_samples_plot']
    plot_data = df[df['d'] == fixed_d].copy()
    plot_data.dropna(subset=['t_sample_mean'], inplace=True)

    if not plot_data.empty:
        ax = sns.lineplot(data=plot_data, x='n_gen', y='t_sample_mean', hue='sampler', marker='o', errorbar='sd')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Number of Samples Generated (log scale)")
        ax.set_ylabel("Mean Sampling Time (s, log scale)")
        ax.set_title(f"Scalability with Sample Count (d = {fixed_d})")
        plt.legend(title='Sampler')
        plt.tight_layout()
        filename = os.path.join(image_dir, "sampling_time_vs_num_samples.pdf")
        plt.savefig(filename, bbox_inches="tight")
        print(f"Saved plot: {filename}")
    else:
        print(f"No data for d={fixed_d}, skipping sample count plot.")
    plt.close()