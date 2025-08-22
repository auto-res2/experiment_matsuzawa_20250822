import yaml
import pandas as pd
import networkx as nx
import os
import time
import math
import torch

from src.preprocess import set_seed, generate_scm_data
from src.train import CausalFactorSampler, McmcSamplerPlaceholder, KernelDensitySamplerPlaceholder
from src.evaluate import benchmark_sampler, choose_intervention_variable, plot_results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_full_experiment(config):
    """Runs the full scalability and speed benchmark experiment."""
    exp_params = config['experiment_params']
    model_params = config['model_params']
    paths = config['paths']

    set_seed(exp_params['seed'])
    results = []

    for d in exp_params['d_values']:
        print(f"\n{'='*20} Running for d = {d} {'='*20}")
        
        # 1. Generate or load data
        graph, data = generate_scm_data(
            n_samples=exp_params['n_training_samples'], 
            d=d, 
            data_dir=paths['data_dir'],
            seed=exp_params['seed']
        )
        intervention_var, intervention_val = choose_intervention_variable(graph, data)

        # 2. Train CausalFactorSampler
        epochs = min(model_params['cfs_epochs_base'], model_params['cfs_epochs_divisor'] // d)
        start_train_time = time.time()
        our_sampler = CausalFactorSampler(
            data, graph=graph, 
            num_latent_dims=math.ceil(d / 10), 
            epochs=epochs,
            batch_size=model_params['cfs_batch_size'],
            lr=model_params['cfs_lr']
        )
        t_train = time.time() - start_train_time
        our_sampler.save_model(os.path.join(paths['model_dir'], f'cfs_model_d{d}.pth'))

        # 3. Instantiate baselines
        mcmc_sampler = McmcSamplerPlaceholder(data, graph=graph)
        kde_sampler = KernelDensitySamplerPlaceholder(data, graph=graph)

        samplers = {
            'CausalFactorSampler': (our_sampler, t_train),
            'McmcSampler': (mcmc_sampler, 0),
            'KernelDensitySampler': (kde_sampler, 0)
        }

        # 4. Run benchmarks
        for n_gen in exp_params['n_samples_to_generate']:
            print(f"--- Benchmarking for N_samples = {n_gen} ---")
            for name, (sampler, train_time) in samplers.items():
                result = benchmark_sampler(
                    sampler, name, n_gen, intervention_var, 
                    intervention_val, exp_params['n_runs'], d, train_time
                )
                results.append(result)

    # 5. Save and plot results
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(paths['results_file']), exist_ok=True)
    results_df.to_csv(paths['results_file'], index=False)
    print(f"\nResults saved to {paths['results_file']}")
    print("\n--- Final Results ---")
    print(results_df)

    plot_results(results_df, paths['image_dir'], config['plotting'])

def test_run():
    """A quick test to ensure all components run without errors."""
    print("\n\nRunning a quick test of the pipeline...")
    try:
        # Setup a minimal config for the test
        d = 10
        n_training = 200
        n_gen = 50
        epochs = 2

        set_seed(42)
        
        print("1. Preprocessing...")
        graph, data = generate_scm_data(n_training, d, 'data', 42)
        
        print("2. Training CausalFactorSampler...")
        cfs = CausalFactorSampler(data, graph, num_latent_dims=math.ceil(d/10), epochs=epochs)
        
        print("3. Evaluating samplers...")
        intervention_var, intervention_val = choose_intervention_variable(graph, data)
        _ = benchmark_sampler(cfs, 'CausalFactorSampler', n_gen, intervention_var, intervention_val, 1, d, 0.1)
        
        mcmc = McmcSamplerPlaceholder(data, graph)
        _ = benchmark_sampler(mcmc, 'McmcSampler', n_gen, intervention_var, intervention_val, 1, d, 0)

        kde = KernelDensitySamplerPlaceholder(data, graph)
        _ = benchmark_sampler(kde, 'KernelDensitySampler', n_gen, intervention_var, intervention_val, 1, d, 0)
        
        print("\nQuick test completed successfully.")
        return True
    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if test_run():
        print(f"\n\n{'*'*20} STARTING FULL EXPERIMENT {'*'*20}")
        print(f"Using device: {DEVICE}")
        print("NOTE: This experiment can take a significant amount of time.")
        run_full_experiment(config)
        print(f"\n{'*'*20} EXPERIMENT FINISHED {'*'*20}")
    else:
        print("\nSkipping full experiment due to test failure.")
