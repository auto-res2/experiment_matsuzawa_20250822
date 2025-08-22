# -*- coding: utf-8 -*-
"""
main.py

Entry point to run the SURGE-Prompt experiments pipeline:
- Preprocess synthetic data
- Train (run experiments) and save figures/models
- Evaluate saved models

Usage examples:
- python src/main.py --quick           # quick functional run (fast)
- python src/main.py --full            # full run (longer)
- python src/main.py --test            # run built-in test to verify outputs

All figures are saved as PDF to .research/iteration1/images.
"""
import os
import argparse
import yaml

from preprocess import preprocess_all
from train import run_experiment_1, run_experiment_2, run_experiment_3, run_all, test_quick_run, IMAGES_DIR_DEFAULT
from evaluate import evaluate_saved_surrogate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH_DEFAULT = os.path.join(BASE_DIR, 'config', 'config.yaml')
IMAGES_DIR = IMAGES_DIR_DEFAULT


def ensure_dirs():
    os.makedirs(os.path.join(BASE_DIR, '.research', 'iteration1', 'images'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)


def load_config(path: str = CONFIG_PATH_DEFAULT):
    if not os.path.exists(path):
        print(f"Config not found at {path}. Using defaults.")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_PATH_DEFAULT)
    parser.add_argument('--quick', action='store_true', help='Run in quick mode (small budgets).')
    parser.add_argument('--full', action='store_true', help='Run full experiments (larger budgets).')
    parser.add_argument('--test', action='store_true', help='Run quick functional test suite.')
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config(args.config)

    seed = int(cfg.get('seed', 0))
    images_dir = cfg.get('paths', {}).get('images_dir', IMAGES_DIR)

    # Preprocess
    print("[Main] Preprocessing synthetic datasets...")
    preprocess_all(seed=seed, data_dir=os.path.join(BASE_DIR, 'data'))

    # Run experiments
    if args.test:
        test_quick_run(images_dir=images_dir)
        print('[Main] Test run completed successfully.')
        return

    if args.full:
        print('[Main] Running full experiments (this may take longer)...')
        run_experiment_1(seed=seed, quick=False, images_dir=images_dir)
        run_experiment_2(seed=seed+1, quick=False, images_dir=images_dir)
        run_experiment_3(seed=seed+2, quick=False, images_dir=images_dir)
    elif args.quick:
        print('[Main] Running quick experiments...')
        run_experiment_1(seed=seed, quick=True, images_dir=images_dir)
        run_experiment_2(seed=seed+1, quick=True, images_dir=images_dir)
        run_experiment_3(seed=seed+2, quick=True, images_dir=images_dir)
    else:
        print('[Main] No explicit mode given; running quick all-in-one and evaluation...')
        run_all(seed=seed, images_dir=images_dir)

    # Evaluate saved surrogate (optional post-run example)
    print('[Main] Evaluating saved surrogate on seed prompt...')
    evaluate_saved_surrogate(seed=seed, images_dir=images_dir)


if __name__ == '__main__':
    main()
