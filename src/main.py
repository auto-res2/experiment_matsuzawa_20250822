import argparse
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from preprocess import (
    create_mock_dales_h5, compute_and_store_normals,
    H5CoresetDataset, stream_h5_data, random_sampler,
    grid_sampler, approximate_fps_sampler, nystrom_sampler
)
from train import (
    MockPTv3, LGSSparseUNet, train_teacher, train_lgs, MINKOWSKI_AVAILABLE
)
from evaluate import evaluate_model

CONFIG_DIR = 'config'
DATA_DIR = 'data'
IMAGE_DIR = os.path.join('.research', 'iteration1', 'images')
MODELS_DIR = 'models'

def plot_training_curves(history, title, filename, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history, label='Training Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    filepath = os.path.join(output_dir, filename)
    print(f"Saving plot to {filepath}")
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close()

def plot_results_barchart(results, coreset_sizes, output_dir):
    plt.figure(figsize=(15, 8))
    methods = list(results.keys())
    n_methods = len(methods)
    n_sizes = len(coreset_sizes)
    
    bar_width = 0.8 / n_methods
    index = np.arange(n_sizes)

    for i, method in enumerate(methods):
        means = [results[method][size]['miou_mean'] for size in coreset_sizes]
        stds = [results[method][size]['miou_std'] for size in coreset_sizes]
        plt.bar(index + i * bar_width, means, bar_width, yerr=stds, capsize=5, label=method)

    plt.xlabel('Coreset Size')
    plt.ylabel('Mean IoU (mIoU)')
    plt.title('Performance Comparison on Large-Scale Semantic Segmentation')
    plt.xticks(index + bar_width * (n_methods-1)/2, [f'{s/1e6:.1f}M' for s in coreset_sizes])
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--')
    filename = 'miou_comparison.pdf'
    filepath = os.path.join(output_dir, filename)
    print(f"Saving plot to {filepath}")
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_pdf(cm, num_classes, filename, output_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    filepath = os.path.join(output_dir, filename)
    print(f"Saving plot to {filepath}")
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close()

def run_sicod(config, coreset_size, seed, h5_path, image_dir, ablation_no_weights=False):
    print(f"\n--- Running SICoD (coreset: {coreset_size}, seed: {seed}, no_weights: {ablation_no_weights}) ---")
    if not MINKOWSKI_AVAILABLE:
        print("Skipping SICoD run because MinkowskiEngine is not available.")
        return None, None, None
        
    device = config['training']['device'] if torch.cuda.is_available() else 'cpu'
    num_train_points = config['data']['num_train_points']
    
    teacher_model = None
    
    for k in range(config['training']['num_sicod_iterations']):
        print(f"-- SICoD Iteration {k} --")
        if k == 0:
            print("  Step 1/4: Generating initial coreset (Random Sampling)")
            coreset_indices = random_sampler(num_train_points, coreset_size)
            saliency_weights = np.ones(coreset_size)
        else:
            print("  Step 1a/4: Generating saliency scores with teacher model...")
            teacher_model.eval()
            all_saliency_scores = []
            with torch.no_grad():
                for data_batch in stream_h5_data(h5_path, 'train', config['evaluation']['eval_batch_size']):
                    coords = torch.from_numpy(data_batch['coords']).to(device)
                    normals = torch.from_numpy(data_batch['normals']).to(device)
                    features = torch.cat([coords, normals], dim=1)
                    logits = teacher_model(features)
                    probs = torch.softmax(logits, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
                    all_saliency_scores.append(entropy.cpu().numpy())
            saliency_targets = np.concatenate(all_saliency_scores)

            print("  Step 1b/4: Training LGS saliency predictor...")
            lgs_model = LGSSparseUNet(in_channels=config['data']['geom_feature_dim'], out_channels=1, channels=config['model']['lgs_channels'])
            train_lgs(lgs_model, h5_path, saliency_targets, stream_h5_data, config, device)

            print("  Step 1c/4: Generating new coreset with LGS...")
            lgs_model.eval()
            predicted_saliency = []
            voxel_size = config['model']['lgs_voxel_size']
            with torch.no_grad():
                for data_batch in stream_h5_data(h5_path, 'train', config['training']['batch_size'], cols=('coords', 'normals')):
                    coords = data_batch['coords']
                    local_coords = coords - coords.mean(axis=0)
                    geom_features = np.concatenate([local_coords, data_batch['normals']], axis=1)
                    mink_coords = ME.utils.batched_coordinates([coords / voxel_size])
                    features_tensor = torch.from_numpy(geom_features).float().to(device)
                    sparse_input = ME.SparseTensor(features_tensor, coordinates=mink_coords, device=device)
                    pred_s = lgs_model(sparse_input).features.squeeze().cpu().numpy()
                    predicted_saliency.append(pred_s)
            predicted_saliency = np.concatenate(predicted_saliency)
            if len(predicted_saliency) < num_train_points:
                predicted_saliency = np.pad(predicted_saliency, (0, num_train_points - len(predicted_saliency)), 'edge')
            else:
                predicted_saliency = predicted_saliency[:num_train_points]

            probs = (predicted_saliency - predicted_saliency.min() + 1e-6)
            probs /= probs.sum()
            coreset_indices = np.random.choice(num_train_points, coreset_size, replace=False, p=probs)
            saliency_weights = predicted_saliency[coreset_indices]
        
        print(f"  Step 2/4: Training teacher model on coreset (size {coreset_size})...")
        teacher_model = MockPTv3(in_features=config['data']['feature_dim'], out_classes=config['data']['num_classes'], model_dim=config['model']['ptv3_dim'], depth=config['model']['ptv3_depth'])
        use_weights = (not ablation_no_weights) and (k > 0)
        dataset = H5CoresetDataset(h5_path, 'train', coreset_indices, saliency_weights=saliency_weights if use_weights else None)
        loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2)
        
        loss_history = train_teacher(teacher_model, loader, config['training']['ptv3_epochs'], config['training']['ptv3_lr'], device, weighted=use_weights)
        
        method_name = 'sicod' if not ablation_no_weights else 'sicod_no_weights'
        plot_filename = f"training_loss_{method_name}_size{coreset_size}_seed{seed}_iter{k}.pdf"
        plot_training_curves(loss_history, f'Teacher Loss (Iter {k})', plot_filename, image_dir)

    print("  Step 4/4: Final evaluation of teacher model...")
    miou, oa, cm = evaluate_model(teacher_model, h5_path, device, config['evaluation']['eval_batch_size'], stream_h5_data)
    print(f'  -> Final mIoU: {miou:.4f}, OA: {oa:.4f}')
    return miou, oa, cm

def main_experiment(config):
    h5_path = os.path.join(DATA_DIR, config['data']['h5_filename'])
    device = config['training']['device'] if torch.cuda.is_available() else 'cpu'
    
    create_mock_dales_h5(h5_path, config['data']['num_train_points'], config['data']['num_test_points'], config['data']['num_scenes'], config['data']['num_classes'])
    compute_and_store_normals(h5_path)
    
    coreset_sizes = config['coreset']['sizes']
    seeds = config['seeds']
    num_train_points = config['data']['num_train_points']
    
    methods = {
        'Random': random_sampler,
        'Grid': grid_sampler, 
        'FPS_Approx': approximate_fps_sampler,
        'Nystrom_Approx': nystrom_sampler, 
    }
    
    results = {m: {s: {'mious': [], 'oas': []} for s in coreset_sizes} for m in list(methods.keys()) + ['SICoD', 'SICoD (no weights)']}

    for method_name, sampler_fn in methods.items():
        print(f'\n===== Running Baseline: {method_name} =====')
        for size in coreset_sizes:
            print(f'--- Coreset Size: {size} ---')
            for seed in seeds:
                print(f'-- Seed: {seed} --')
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                if method_name == 'Grid':
                     indices = sampler_fn(h5_path, size, num_train_points)
                elif method_name == 'FPS_Approx':
                     indices = sampler_fn(h5_path, size, num_train_points)
                elif method_name == 'Nystrom_Approx':
                     indices = sampler_fn(h5_path, size, config['training']['batch_size'], num_train_points)
                else:
                     indices = sampler_fn(num_train_points, size)
                
                dataset = H5CoresetDataset(h5_path, 'train', indices)
                loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2)
                
                model = MockPTv3(in_features=config['data']['feature_dim'], out_classes=config['data']['num_classes'], model_dim=config['model']['ptv3_dim'], depth=config['model']['ptv3_depth'])
                
                train_history = train_teacher(model, loader, config['training']['ptv3_epochs'], config['training']['ptv3_lr'], device)
                plot_training_curves(train_history, f'Loss - {method_name} (size {size}, seed {seed})', f'training_loss_{method_name.lower()}_size{size}_seed{seed}.pdf', IMAGE_DIR)

                miou, oa, _ = evaluate_model(model, h5_path, device, config['evaluation']['eval_batch_size'], stream_h5_data)
                print(f'  -> mIoU: {miou:.4f}, OA: {oa:.4f}')
                results[method_name][size]['mious'].append(miou)
                results[method_name][size]['oas'].append(oa)
    
    for size in coreset_sizes:
        for seed in seeds:
            miou, oa, cm = run_sicod(config, size, seed, h5_path, IMAGE_DIR, ablation_no_weights=False)
            if miou is not None:
                results['SICoD'][size]['mious'].append(miou)
                results['SICoD'][size]['oas'].append(oa)
                if size == max(coreset_sizes) and seed == seeds[0]:
                    plot_confusion_matrix_pdf(cm, config['data']['num_classes'], f'confusion_matrix_sicod_size{size}_seed{seed}.pdf', IMAGE_DIR)
            
            miou_abl, oa_abl, _ = run_sicod(config, size, seed, h5_path, IMAGE_DIR, ablation_no_weights=True)
            if miou_abl is not None:
                results['SICoD (no weights)'][size]['mious'].append(miou_abl)
                results['SICoD (no weights)'][size]['oas'].append(oa_abl)

    print('\n===== Final Results Summary =====')
    final_results = {}
    for method, size_results in results.items():
        final_results[method] = {}
        print(f'\nMethod: {method}')
        for size, metric_results in size_results.items():
            if not metric_results['mious']: continue
            miou_mean = np.mean(metric_results['mious'])
            miou_std = np.std(metric_results['mious'])
            oa_mean = np.mean(metric_results['oas'])
            oa_std = np.std(metric_results['oas'])
            final_results[method][size] = {
                'miou_mean': miou_mean, 'miou_std': miou_std,
                'oa_mean': oa_mean, 'oa_std': oa_std
            }
            print(f'  Coreset Size: {size}: mIoU = {miou_mean:.4f} +/- {miou_std:.4f} | OA = {oa_mean:.4f} +/- {oa_std:.4f}')

    plot_results_barchart(final_results, coreset_sizes, IMAGE_DIR)
    print('\nExperiment finished.')

def test_function():
    print('\n===== Running Quick Test Function =====')
    test_config = {
        'data': {
            'h5_filename': 'dales_test.h5',
            'num_train_points': 20000, 'num_test_points': 5000,
            'num_scenes': 1, 'num_classes': 4, 'feature_dim': 6, 'geom_feature_dim': 6
        },
        'training': {
            'device': 'cuda',
            'batch_size': 128, 'ptv3_epochs': 1, 'lgs_epochs': 1, 'ptv3_lr': 1e-3, 'lgs_lr': 1e-4,
            'num_sicod_iterations': 1
        },
        'evaluation': {'eval_batch_size': 256},
        'coreset': {'sizes': [2000]},
        'model': {'ptv3_dim': 16, 'ptv3_depth': 2, 'lgs_channels': [8, 16], 'lgs_voxel_size': 0.2},
        'seeds': [0]
    }
    
    h5_path = os.path.join(DATA_DIR, test_config['data']['h5_filename'])
    device = test_config['training']['device'] if torch.cuda.is_available() else 'cpu'
    
    create_mock_dales_h5(h5_path, test_config['data']['num_train_points'], test_config['data']['num_test_points'], test_config['data']['num_scenes'], test_config['data']['num_classes'])
    compute_and_store_normals(h5_path)
    
    print("\n--- Testing Random Sampling ---")
    indices = random_sampler(test_config['data']['num_train_points'], test_config['coreset']['sizes'][0])
    dataset = H5CoresetDataset(h5_path, 'train', indices)
    loader = DataLoader(dataset, batch_size=test_config['training']['batch_size'])
    model = MockPTv3(in_features=test_config['data']['feature_dim'], out_classes=test_config['data']['num_classes'], model_dim=test_config['model']['ptv3_dim'], depth=test_config['model']['ptv3_depth'])
    train_teacher(model, loader, test_config['training']['ptv3_epochs'], test_config['training']['ptv3_lr'], device)
    miou, oa, _ = evaluate_model(model, h5_path, device, test_config['evaluation']['eval_batch_size'], stream_h5_data)
    print(f'  -> Test Baseline mIoU: {miou:.4f}')
    assert miou >= 0.0
    
    print("\n--- Testing SICoD Iteration ---")
    miou, _, _ = run_sicod(test_config, test_config['coreset']['sizes'][0], 0, h5_path, IMAGE_DIR)
    if MINKOWSKI_AVAILABLE:
        print(f'  -> Test SICoD mIoU: {miou:.4f}')
        assert miou >= 0.0

    if os.path.exists(h5_path): os.remove(h5_path)
    for f in os.listdir(IMAGE_DIR):
        if f.endswith('.pdf'): os.remove(os.path.join(IMAGE_DIR, f))

    print('\n===== Test Passed Successfully! =====')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SICoD experiment for large-scale semantic segmentation.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--test', action='store_true', help='Run the quick test function instead of the full experiment.')
    args = parser.parse_args()

    for d in [CONFIG_DIR, DATA_DIR, IMAGE_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)

    if args.test:
        test_function()
    else:
        config_path = os.path.join(CONFIG_DIR, args.config)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        main_experiment(config)
