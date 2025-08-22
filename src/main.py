import torch
import numpy as np
import os
import json
from .train_simple import train_aofm
from .evaluate import evaluate_model, plot_loss_curves

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_config():
    """Get experiment configuration optimized for Tesla T4 GPU."""
    config = {
        'experiment_name': 'AOFM_FID_BENCHMARK_CIFAR10',
        'dataset': 'CIFAR10',
        'img_size': 32,
        'img_channels': 3,
        'batch_size': 8,  # Very small for testing
        'num_iterations': 50,  # Very short for testing
        'lr': 1e-4,
        'warmup_steps': 5,
        'lambda_inv': 1.0,
        'cg_iters': 10,
        'num_workers': 2,
        'seed': 42,
        'output_dir': './models',
        'images_dir': '.research/iteration1/images',
        'fid_samples': 50000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    return config

def update_status(status):
    """Update experiment status."""
    status_file = '.research/status.json'
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    
    status_data = {
        'status_enum': status,
        'timestamp': str(torch.tensor(0).item())  # Simple timestamp
    }
    
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    print(f'[INFO] Status updated to: {status}')

def main():
    """Main experiment execution function."""
    print('='*60)
    print('AOFM (Amortized Optimal Flow Matching) Experiment')
    print('CIFAR-10 FID Benchmark')
    print('='*60)
    
    config = get_config()
    set_seed(config['seed'])
    
    print(f'[INFO] Device: {config["device"]}')
    print(f'[INFO] Batch size: {config["batch_size"]}')
    print(f'[INFO] Training iterations: {config["num_iterations"]}')
    print(f'[INFO] Output directory: {config["output_dir"]}')
    print(f'[INFO] Images directory: {config["images_dir"]}')
    
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['images_dir'], exist_ok=True)
    
    try:
        update_status('running')
        
        print('\n[PHASE 1] Training AOFM Model...')
        models, losses = train_aofm(config)
        
        print('\n[PHASE 2] Plotting Training Curves...')
        loss_plot_filename = os.path.join(config['images_dir'], 'training_losses.pdf')
        plot_loss_curves(losses, loss_plot_filename)
        
        print('\n[PHASE 3] Evaluating Model...')
        results = evaluate_model(models, config)
        
        print('\n' + '='*60)
        print('EXPERIMENT RESULTS SUMMARY')
        print('='*60)
        print(f'Training completed: {config["num_iterations"]} iterations')
        print(f'Final losses:')
        print(f'  - Total: {losses["total"][-1]:.4f}')
        print(f'  - OFM: {losses["ofm"][-1]:.4f}')
        print(f'  - Inverter: {losses["inv"][-1]:.4f}')
        
        print(f'\nFID Scores (Placeholder):')
        for nfe, fid in results['fid_scores'].items():
            print(f'  - NFE={nfe}: {fid:.1f}')
        
        print(f'\nGenerated files:')
        print(f'  - Models: {config["output_dir"]}/icnn_final.pth, inverter_final.pth')
        print(f'  - Loss curves: {loss_plot_filename}')
        print(f'  - FID plot: {results["fid_plot"]}')
        for nfe in [1, 2, 4, 10, 20]:
            print(f'  - Samples NFE={nfe}: {results[f"nfe_{nfe}"]["filename"]}')
        
        print('\n[SUCCESS] AOFM experiment completed successfully!')
        
        update_status('stopped')
        
    except Exception as e:
        print(f'\n[ERROR] Experiment failed: {str(e)}')
        update_status('failed')
        raise e
    
    print('='*60)

if __name__ == '__main__':
    main()
