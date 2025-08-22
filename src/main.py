import os
import yaml
import argparse
import pandas as pd
import numpy as np
import torch
from diffusers import DDPMPipeline

from preprocess import get_cifar10_dataloaders, generate_synthetic_data
from train import train_classifier
from evaluate import calculate_fid, plot_training_curves, plot_summary_results

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config):
    """Main function to run the experiment."""
    # Create necessary directories
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(config['DATA_DIR'], exist_ok=True)
    os.makedirs(config['MODELS_DIR'], exist_ok=True)

    print(f"Using device: {config['DEVICE']}")
    results = []

    # --- Phase 0: Real Data (Upper Bound) ---
    real_train_set, real_test_loader = get_cifar10_dataloaders(config)
    real_data_accuracies = []
    history_real_data = None
    for seed in range(config['NUM_SEEDS']):
        acc, hist = train_classifier(real_train_set, real_test_loader, "Real Data", seed, config)
        real_data_accuracies.append(acc)
        if seed == 0: history_real_data = hist
    
    results.append({
        'Method': 'Real Data',
        'Generation Time (ms/image)': 0,
        'FID': 0,
        'Accuracy': np.mean(real_data_accuracies),
        'Accuracy Std': np.std(real_data_accuracies)
    })
    plot_training_curves(history_real_data, 'real_data', config)

    # --- Phases 1 & 2: Generation and Evaluation for Synthetic Methods ---
    try:
        pipeline = DDPMPipeline.from_pretrained(config['DIFFUSION_MODEL_ID'])
    except Exception as e:
        print(f"Could not load diffusion model. Error: {e}")
        return
    
    methods = ['DCD-PCE', 'DistDiff', 'Real-Fake', 'Standard Diffusion']
    for method in methods:
        generated_dataset, gen_time = generate_synthetic_data(pipeline, method, config)
        fid_score = calculate_fid(generated_dataset, method, config)

        accuracies = []
        history_first_run = None
        for seed in range(config['NUM_SEEDS']):
            acc, hist = train_classifier(generated_dataset, real_test_loader, method, seed, config)
            accuracies.append(acc)
            if seed == 0: history_first_run = hist
        
        results.append({
            'Method': method,
            'Generation Time (ms/image)': gen_time,
            'FID': fid_score,
            'Accuracy': np.mean(accuracies),
            'Accuracy Std': np.std(accuracies)
        })
        plot_training_curves(history_first_run, method.lower().replace(' ', '_'), config)

    # --- Phase 3: Results Aggregation and Plotting ---
    results_df = pd.DataFrame(results).set_index('Method')
    print("\n\n" + "="*60)
    print("                  FINAL EXPERIMENT RESULTS")
    print("="*60)
    print(results_df.to_string(float_format="%.2f"))
    print("="*60)

    plot_summary_results(results_df, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the DCD-PCE comparative experiment.")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file.')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with fewer images and epochs.')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.test_mode:
        print("\n" + "*"*40)
        print("           RUNNING IN TEST MODE")
        print("*"*40 + "\n")
        config['NUM_GEN_IMAGES'] = config['TEST_NUM_GEN_IMAGES']
        config['TRAIN_EPOCHS'] = config['TEST_TRAIN_EPOCHS']
        config['NUM_SEEDS'] = 1
    
    # Set device
    config['DEVICE'] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config)
