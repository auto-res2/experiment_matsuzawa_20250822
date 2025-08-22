import torch
import torch.optim as optim
import yaml
import argparse
import random
import numpy as np
import os
from copy import deepcopy

# Import from our source files
from preprocess import get_dataset
from train import MetaLearner, train_epoch
from evaluate import evaluate, analyze_and_plot_results

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(config_path):
    # --- 1. Load Configuration & Setup ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    use_cuda = not config['system']['no_cuda'] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device: {device}')

    # Create directories
    os.makedirs(config['experiment']['results_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['experiment']['results_dir'], 'images'), exist_ok=True)
    os.makedirs(config['experiment']['models_dir'], exist_ok=True)

    dataset_name = f"{config['experiment']['dataset'].capitalize()} {config['experiment']['n_way']}-way {config['experiment']['k_shot']}-shot"
    print(f"--- Starting Experiment: {dataset_name} ---")

    all_final_results = {}
    
    # --- 2. Load Data ---
    # Data is loaded once for all runs to avoid repeated downloads
    train_loader, _, test_loader = get_dataset(config)

    # --- 3. Run Experiment Loop ---
    algorithms_to_run = config['experiment']['run_algorithms']
    
    for algorithm in algorithms_to_run:
        print(f"\n{'='*20} RUNNING: {algorithm.upper()} {'='*20}")
        
        # Create a copy of the config for this specific algorithm run
        run_config = deepcopy(config)
        run_config['experiment']['algorithm'] = algorithm
        
        seed_results = []
        for seed in range(config['experiment']['num_seeds']):
            print(f"\n--- Running Seed {seed + 1}/{config['experiment']['num_seeds']} for {algorithm.upper()} ---")
            set_seed(seed)

            # --- Model & Optimizer ---
            model = MetaLearner(run_config).to(device)
            optimizer = optim.Adam(model.parameters(), lr=run_config['training']['meta_lr'])

            # --- Meta-Training Loop ---
            for epoch in range(run_config['training']['num_epochs']):
                print(f"Epoch {epoch + 1}/{run_config['training']['num_epochs']}")
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, run_config, device)
                print(f"Epoch {epoch + 1} Summary: Avg Loss = {train_loss:.4f}, Avg Acc = {train_acc:.4f}")
            
            # --- Save Model ---
            model_save_path = os.path.join(run_config['experiment']['models_dir'], f'{algorithm}_{seed}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved trained model to {model_save_path}")

            # --- Meta-Evaluation ---
            # For evaluation, we might need a model with the original num_inner_steps
            eval_config = deepcopy(run_config)
            if algorithm == 'veloml':
                # Evaluate with fixed steps like baselines
                 eval_config['training']['num_inner_steps'] = config['training']['num_inner_steps']
            
            results = evaluate(model, test_loader, eval_config, device)
            seed_results.append(results)

        all_final_results[algorithm] = seed_results

    # --- 4. Final Analysis & Plotting ---
    analyze_and_plot_results(all_final_results, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VeloML experiments from a config file.')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    main(args.config)
