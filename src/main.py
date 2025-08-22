import os
import yaml
import pandas as pd
import numpy as np
import torch
import argparse

from src.preprocess import create_mock_dataloader
from src.train import get_models
from src.evaluate import evaluate_model, plot_pareto_frontier, plot_budget_adherence, plot_cost_consistency

def run_experiment(config, test_run=False):
    """Main function to run the entire evaluation suite."""
    
    exp_cfg = config['experiment']
    data_cfg = config['data']
    eval_cfg = config['evaluation']
    
    print(f"--- Starting Experiment: {exp_cfg['name']} ---")
    
    # --- Setup Environment ---
    if exp_cfg['device'] == 'auto':
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(exp_cfg['device'])
    print(f"Using device: {DEVICE}")
    
    FIGURE_DIR = exp_cfg['figure_dir']
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
        print(f"Created directory: {FIGURE_DIR}")

    # --- Setup Dataset ---
    num_samples = data_cfg['batch_size'] * 2 if test_run else data_cfg['num_samples']
    val_loader = create_mock_dataloader(num_samples=num_samples, batch_size=data_cfg['batch_size'])
    print("Dataset prepared.")

    # --- Setup Models ---
    metabapl_model, model_zoo, policy_zoo, confidence_model = get_models(config)
    
    all_results = []

    # --- Evaluation: Meta-BAPL ---
    print("\nEvaluating Meta-BAPL model...")
    # Grid points
    lat_grid = eval_cfg['metabapl_budgets']['latency_grid']
    gflops_grid = eval_cfg['metabapl_budgets']['gflops_grid']
    for lat in lat_grid:
        for gflops in gflops_grid:
            cfg = {'name': 'Meta-BAPL', 'type': 'metabapl', 'budget': [lat, gflops]}
            all_results.append(evaluate_model(metabapl_model, val_loader, cfg, DEVICE))
    # Interpolated points
    for _ in range(eval_cfg['metabapl_budgets']['interp_points']):
        lat = np.random.uniform(min(lat_grid), max(lat_grid))
        gflops = np.random.uniform(min(gflops_grid), max(gflops_grid))
        cfg = {'name': 'Meta-BAPL (Interp)', 'type': 'metabapl_interp', 'budget': [lat, gflops]}
        all_results.append(evaluate_model(metabapl_model, val_loader, cfg, DEVICE))

    # --- Evaluation: Baselines ---
    print("\nEvaluating baselines...")
    # Model Zoo
    for name, model in model_zoo.items():
        cfg = {'name': name, 'type': 'model_zoo'}
        all_results.append(evaluate_model(model, val_loader, cfg, DEVICE))
    # Policy Zoo
    for name, model in policy_zoo.items():
        cfg = {'name': name, 'type': 'policy_zoo'}
        all_results.append(evaluate_model(model, val_loader, cfg, DEVICE))
    # Confidence-based
    conf_cfg = eval_cfg['confidence_thresholds']
    for threshold in np.linspace(conf_cfg['start'], conf_cfg['end'], conf_cfg['num']):
        cfg = {'name': 'Confidence-based', 'type': 'confidence', 'threshold': threshold}
        all_results.append(evaluate_model(confidence_model, val_loader, cfg, DEVICE))

    # --- Analysis and Visualization ---
    results_df = pd.DataFrame(all_results)
    # Sort for clean plots
    results_df.loc[results_df['type']=='metabapl'] = results_df.loc[results_df['type']=='metabapl'].sort_values('gflops')
    results_df.loc[results_df['type']=='confidence'] = results_df.loc[results_df['type']=='confidence'].sort_values('gflops')

    print("\n--- Experiment Complete ---")
    print("Final Results Summary (stdout):")
    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(results_df.drop(columns=['per_batch_gflops']).to_string())
    
    if not test_run:
        plot_pareto_frontier(results_df, FIGURE_DIR)
        plot_budget_adherence(results_df, FIGURE_DIR)
        plot_cost_consistency(results_df, FIGURE_DIR)
    else:
        print("\nSkipping plot generation in test mode.")
    
    print("\nExperiment script finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Meta-BAPL evaluation experiment.")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file.')
    parser.add_argument('--test', action='store_true', help='Run in a quick test mode.')
    args = parser.parse_args()

    print("Loading configuration from:", args.config)
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit(1)

    if args.test:
        print("\n--- Running in TEST mode ---")
    
    run_experiment(config, test_run=args.test)
