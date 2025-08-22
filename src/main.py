import os
import argparse
import yaml
import numpy as np

from src.preprocess import preprocess
from src.train import train_experiment_1, train_experiment_2, train_experiment_3
from src.evaluate import evaluate_exp1, evaluate_exp2, evaluate_exp3

# Directories per spec
RESEARCH_IMG_DIR = os.path.join('.research', 'iteration1', 'images')
CONFIG_DIR_DEFAULT = 'config'
DATA_DIR_DEFAULT = 'data'
MODELS_DIR_DEFAULT = 'models'


def parse_args():
    ap = argparse.ArgumentParser(description='SAMC2 Experiments Runner')
    ap.add_argument('--config', type=str, default=os.path.join(CONFIG_DIR_DEFAULT, 'config.yaml'), help='Path to config YAML')
    ap.add_argument('--experiment', type=str, default=None, help='Override experiment: test|exp1|exp2|exp3')
    ap.add_argument('--quick_test', action='store_true', help='Run a minimal quick test regardless of config')
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.experiment is not None:
        config['experiment'] = args.experiment

    os.makedirs(RESEARCH_IMG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR_DEFAULT, exist_ok=True)
    os.makedirs(MODELS_DIR_DEFAULT, exist_ok=True)

    # Quick test overrides
    if args.quick_test:
        config['experiment'] = 'test'
        config['T'] = 2000
        config['seed'] = 0
        config['alpha'] = 0.1
        config['rsc_windows'] = [64, 128, 256]
        print('[MAIN] Running quick test...')

    meta = preprocess(config, DATA_DIR_DEFAULT)
    data_path = meta['data_path']
    alpha = float(config.get('alpha', 0.1))

    model_paths = []
    exp = str(config.get('experiment', 'test')).lower()

    if exp == 'exp1' or (args.quick_test and exp == 'test'):
        d = np.load(data_path, allow_pickle=True)
        y = d['y']
        seeds = config.get('seeds', [int(config.get('seed', 0))])
        rsc_windows = tuple(config.get('rsc_windows', [128, 512, 2048]))
        print(f"[MAIN] Training Exp1 with seeds={seeds}, alpha={alpha}, rsc_windows={rsc_windows}")
        model_paths = train_experiment_1(y, MODELS_DIR_DEFAULT, alpha, seeds, rsc_windows)
        evaluate_exp1(model_paths, RESEARCH_IMG_DIR, alpha)
    elif exp == 'exp2':
        d = np.load(data_path, allow_pickle=True)
        y = d['y']
        rsc_windows = tuple(config.get('rsc_windows', [168, 672, 2688]))
        print(f"[MAIN] Training Exp2 (electricity) with alpha={alpha}, rsc_windows={rsc_windows}")
        model_paths = train_experiment_2(y, MODELS_DIR_DEFAULT, alpha, rsc_windows)
        evaluate_exp2(model_paths[0], RESEARCH_IMG_DIR, alpha)
    elif exp == 'exp3':
        d = np.load(data_path, allow_pickle=True)
        y = d['y']
        print(f"[MAIN] Training Exp3 (ablation) with alpha={alpha}")
        model_paths = train_experiment_3(y, MODELS_DIR_DEFAULT, alpha)
        evaluate_exp3(model_paths, RESEARCH_IMG_DIR)
    else:  # default test pipeline uses exp1 small
        d = np.load(data_path, allow_pickle=True)
        y = d['y']
        seeds = [int(config.get('seed', 0))]
        rsc_windows = tuple(config.get('rsc_windows', [64, 128, 256]))
        print(f"[MAIN] Training TEST (small Exp1) with alpha={alpha}, rsc_windows={rsc_windows}")
        model_paths = train_experiment_1(y, MODELS_DIR_DEFAULT, alpha, seeds, rsc_windows)
        evaluate_exp1(model_paths, RESEARCH_IMG_DIR, alpha)

    print('[MAIN] Done. Figures saved under .research/iteration1/images and results saved under models/.')


if __name__ == '__main__':
    main()
