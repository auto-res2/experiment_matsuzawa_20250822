import os
import yaml
import pandas as pd
from pathlib import Path
import sys

# Add src to path to allow for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import preprocess_data
from src.train import run_hpo_trial, RandomSearch, ASHA, BOHB, CAR_HPO, DyHPO
from src.evaluate import plot_curves, analyze_results

def main():
    """Main script to run the HPO experiment."""
    print("--- Starting HPO Experiment ---")

    # 1. Load Configuration
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        exp_config = config['experiment']
        benchmarks = config['benchmarks']
        optimizers_config = config['optimizers']
    except FileNotFoundError:
        print("Error: config/config.yaml not found. Please ensure it exists.")
        return
    except Exception as e:
        print(f"Error loading or parsing config file: {e}")
        return
    
    # 2. Setup
    output_dir = Path(exp_config['output_image_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory for images at: {output_dir}")

    results_csv_path = exp_config['results_csv_path']
    Path(results_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # 3. Preprocessing (Symbolic call as per this experiment's needs)
    preprocess_data()

    # 4. Run HPO Trials
    all_results = []
    optimizer_map = {
        "RandomSearch": RandomSearch, 
        "ASHA": ASHA, 
        "BOHB": BOHB, 
        "CAR-HPO": CAR_HPO,
        "DyHPO": DyHPO
    }
    
    enabled_optimizers = {name: cls for name, cls in optimizer_map.items() if optimizers_config.get(name, False)}
    if not enabled_optimizers:
        print("No optimizers are enabled in config.yaml. Exiting.")
        return

    print(f"Enabled optimizers: {list(enabled_optimizers.keys())}")
    
    for opt_name, opt_class in enabled_optimizers.items():
        for bench_name in benchmarks:
            for seed in range(exp_config['n_seeds']):
                print(f"\nRunning {opt_name} on {bench_name} with seed {seed}...")
                trial_df = run_hpo_trial(
                    optimizer_class=opt_class, 
                    benchmark_name=bench_name, 
                    n_workers=exp_config['n_workers'],
                    wall_clock_limit_secs=exp_config['time_limit_secs'],
                    seed=seed
                )
                if not trial_df.empty:
                    trial_df['optimizer'] = opt_name
                    trial_df['benchmark'] = bench_name
                    trial_df['seed'] = seed
                    all_results.append(trial_df)
    
    if not all_results:
        print("\nExperiment failed to generate any results. Please check your HPOBench and container setup (e.g., Docker or Singularity).")
        return

    final_results_df = pd.concat(all_results, ignore_index=True)
    final_results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved all experiment results to {results_csv_path}")

    # 5. Evaluate Results
    print("\n--- Generating Plots and Analysis ---")
    analyze_results(final_results_df, list(enabled_optimizers.keys()), exp_config['time_limit_secs'])
    
    for bench_name in benchmarks:
        plot_curves(final_results_df, list(enabled_optimizers.keys()), bench_name, exp_config['time_limit_secs'], str(output_dir))

    print("\n--- Experiment Finished Successfully ---")

if __name__ == '__main__':
    # It is assumed this script is run from the root directory of the project.
    # e.g., python src/main.py
    main()
