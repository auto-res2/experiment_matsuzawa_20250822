import os
import argparse

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_DIR = os.path.join(PROJECT_DIR, 'config')

from preprocess import main as preprocess_main
from train import main as train_main
from evaluate import main as evaluate_main


def run_all(config_path: str, force_preprocess: bool):
    print("=== Q-SHIFT Toy Pipeline: preprocess -> train -> evaluate ===")
    preprocess_main(config_path, force=force_preprocess)
    train_main(config_path)
    evaluate_main(config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Q-SHIFT toy research pipeline')
    parser.add_argument('--config', type=str, default=os.path.join(CONFIG_DIR, 'default.yaml'), help='Path to YAML config')
    parser.add_argument('--force-preprocess', action='store_true', help='Regenerate dataset')
    args = parser.parse_args()
    run_all(args.config, args.force_preprocess)
