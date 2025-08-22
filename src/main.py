import os
import argparse
import yaml

from preprocess import preprocess_dataset
from train import train_migad
from evaluate import evaluate_migad


def ensure_all_dirs():
    os.makedirs('.research/iteration1/images', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('config', exist_ok=True)


def run_pipeline(config_path: str):
    ensure_all_dirs()
    print('[Main] Using config:', config_path)

    # 1) Preprocess
    preprocess_dataset(config_path)

    # 2) Train
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    best = train_migad(config)

    # 3) Evaluate
    evaluate_migad(config)

    print('[Main] Pipeline completed successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MiGAD experiment pipeline')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to YAML config')
    args = parser.parse_args()
    run_pipeline(args.config)
