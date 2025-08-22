import os
import yaml
import argparse
import numpy as np
import torch

from src.preprocess import get_dataloaders
from src.train import run_pretraining, run_linear_training
from src.evaluate import evaluate_linear_one_epoch, plot_results, print_summary

def load_config(config_path, test_mode):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg['TEST_MODE'] = test_mode
    
    # Dynamically adjust parameters based on test_mode
    if test_mode:
        print("--- RUNNING IN TEST MODE ---")
        cfg['PRETRAIN_EPOCHS'] = cfg['PRETRAIN_EPOCHS_TEST']
        cfg['PRETRAIN_BATCH_SIZE'] = cfg['PRETRAIN_BATCH_SIZE_TEST']
        cfg['LINEAR_EPOCHS'] = cfg['LINEAR_EPOCHS_TEST']
        cfg['LINEAR_BATCH_SIZE'] = cfg['LINEAR_BATCH_SIZE_TEST']
        cfg['MOCO_K'] = cfg['MOCO_K_TEST']
        cfg['MTS_K_HARD'] = cfg['MTS_K_HARD_TEST']
    else:
        print("--- RUNNING IN FULL EXPERIMENT MODE ---")
        cfg['PRETRAIN_EPOCHS'] = cfg['PRETRAIN_EPOCHS_FULL']
        cfg['PRETRAIN_BATCH_SIZE'] = cfg['PRETRAIN_BATCH_SIZE_FULL']
        cfg['LINEAR_EPOCHS'] = cfg['LINEAR_EPOCHS_FULL']
        cfg['LINEAR_BATCH_SIZE'] = cfg['LINEAR_BATCH_SIZE_FULL']
        cfg['MOCO_K'] = cfg['MOCO_K_FULL']
        cfg['MTS_K_HARD'] = cfg['MTS_K_HARD_FULL']

    cfg['DEVICE'] = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg

def main(args):
    config = load_config(args.config, args.test_mode)
    
    print(f"Experiment starting. Device: {config['DEVICE']}")
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(os.path.join(config['OUTPUT_DIR'], 'images'), exist_ok=True)
    os.makedirs(config['MODELS_DIR'], exist_ok=True)
    os.makedirs(config['DATA_DIR'], exist_ok=True)

    np.random.seed(config['SEED'])
    torch.manual_seed(config['SEED'])
    if config['DEVICE'] == 'cuda':
        torch.cuda.manual_seed(config['SEED'])

    pretrain_loader, linear_train_loader, linear_test_loader = get_dataloaders(config)
    
    all_results = {}
    pretrain_methods = ['moco', 'dcl', 'srns', 'mts']

    # Run pre-training for all SSL methods
    for method in pretrain_methods:
        checkpoint_path, pretrain_loss = run_pretraining(config, pretrain_loader, method)
        all_results[method] = {'checkpoint': checkpoint_path, 'pretrain_loss': pretrain_loss}

    # Run linear evaluation for all methods
    eval_methods = ['supervised'] + pretrain_methods
    for method in eval_methods:
        pretrained_path = all_results[method]['checkpoint'] if method != 'supervised' else None
        linear_history = run_linear_training(config, linear_train_loader, linear_test_loader, pretrained_path, method, evaluate_linear_one_epoch)
        if method in all_results: 
            all_results[method]['linear_eval'] = linear_history
        else:
            all_results[method] = {'linear_eval': linear_history}

    # Generate plots and print summary
    plot_results(all_results, config)
    print_summary(all_results)

    print("\n--- Experiment Finished Successfully ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Momentum-Based Trajectory Debiasing Experiment')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file.')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with smaller datasets and fewer epochs.')
    
    args = parser.parse_args()
    main(args)
