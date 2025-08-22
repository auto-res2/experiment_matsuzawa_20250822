import os
import yaml
import torch

from train import (
    ensure_dir, set_seed, get_device,
    QToyCNN, build_synthetic_dataset,
    learn_P, fit_S_group,
    collect_bias_bn_params_toy, collect_quant_params_toy
)
from evaluate import save_sketch_package


def preprocess_vision(cfg):
    device = get_device(cfg.get('device'))
    set_seed(int(cfg.get('seed', 1234)))

    models_dir = cfg.get('models_dir', 'models')
    ensure_dir(models_dir)

    C = int(cfg.get('num_classes', 10))

    # Load pretrained model
    model_path = os.path.join(models_dir, 'qtcnn_vision.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Pretrained model not found at {model_path}. Run train.py first.')

    model = QToyCNN(in_ch=3, num_classes=C).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Proxy data for offline fitting
    proxy_data = build_synthetic_dataset(N=200, C=C, patterns=['clean'])

    # Learn P and fit S for groups
    r_e = int(cfg.get('precompute', {}).get('r_e', 8))
    lam = float(cfg.get('precompute', {}).get('ridge_lambda', 1e-2))

    print('[Preprocess] Learning P ...')
    P = learn_P(model, proxy_data, C=C, r_e=r_e, device=device)

    bias_bn_params = collect_bias_bn_params_toy(model)
    quant_params = collect_quant_params_toy(model)

    print('[Preprocess] Fitting S for bias_bn (fp mode) ...')
    S_bias = fit_S_group(model, proxy_data, P, bias_bn_params, lam=lam, mode='fp', device=device)
    print('[Preprocess] Fitting S for quant (quant mode) ...')
    S_quant = fit_S_group(model, proxy_data, P, quant_params, lam=lam, mode='quant', device=device)

    # Save sketch package
    package_path = os.path.join(models_dir, 'sketchbank_vision.pt')
    save_sketch_package(package_path, P, {'bias_bn': [S_bias], 'quant': [S_quant]})
    print(f'[Preprocess] Saved sketch package to {package_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SketchBank - Preprocess (offline P & S fitting)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    preprocess_vision(cfg)
