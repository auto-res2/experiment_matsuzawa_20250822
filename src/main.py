import os
import argparse
import yaml

from .train import run_experiment1, run_experiment2, run_experiment3
from .preprocess import save_datasets


def project_paths():
    src_dir = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(src_dir, '..'))
    research_dir = os.path.join(root, '.research', 'iteration1')
    images_dir = os.path.join(research_dir, 'images')
    config_dir = os.path.join(root, 'config')
    data_dir = os.path.join(root, 'data')
    models_dir = os.path.join(root, 'models')
    logs_dir = research_dir
    for d in [images_dir, config_dir, data_dir, models_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)
    return {
        'root': root,
        'images': images_dir,
        'config': config_dir,
        'data': data_dir,
        'models': models_dir,
        'logs': logs_dir,
    }


def load_config(cfg_path: str):
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    paths = project_paths()
    parser = argparse.ArgumentParser(description='CAMLTO-Delta Experiments Runner')
    parser.add_argument('--config', type=str, default=os.path.join(paths['config'], 'config.yaml'), help='Path to config YAML')
    parser.add_argument('--quick_test', action='store_true', help='Run quick tests for all experiments')
    parser.add_argument('--exp', type=int, default=0, help='Run a specific experiment: 1, 2, or 3')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Always ensure datasets exist (optional saving)
    save_datasets(paths['data'], seed=cfg.get('seed', 0), train_n=cfg.get('train_n', 2400), val_n=cfg.get('val_n', 400))

    if args.quick_test or args.exp == 0:
        print('[Main] Running quick test across all experiments...')
        e1 = cfg.get('experiment1', {})
        run_experiment1(img_dir=paths['images'], log_dir=paths['logs'], models_dir=paths['models'],
                        hours=e1.get('hours', 4), seed=e1.get('seed', 0), carbon_budget=e1.get('carbon_budget', 1.0),
                        steps_per_hour=e1.get('steps_per_hour', 40), csv_trace=e1.get('csv_trace', None), save_prefix=e1.get('save_prefix', 'exp1'))
        e2 = cfg.get('experiment2', {})
        run_experiment2(img_dir=paths['images'], log_dir=paths['logs'], models_dir=paths['models'],
                        hours=e2.get('hours', 6), seed=e2.get('seed', 1), steps_per_hour=e2.get('steps_per_hour', 40),
                        csv_trace=e2.get('csv_trace', None), save_prefix=e2.get('save_prefix', 'exp2'))
        e3 = cfg.get('experiment3', {})
        run_experiment3(img_dir=paths['images'], log_dir=paths['logs'], models_dir=paths['models'],
                        hours=e3.get('hours', 16), seed=e3.get('seed', 2), slots=e3.get('slots', 2),
                        csv_trace=e3.get('csv_trace', None), save_prefix=e3.get('save_prefix', 'exp3'))
        print('[Main] Quick test completed. Check .research/iteration1 for logs and images.')
        return

    if args.exp == 1:
        e1 = cfg.get('experiment1', {})
        run_experiment1(img_dir=paths['images'], log_dir=paths['logs'], models_dir=paths['models'],
                        hours=e1.get('hours', 6), seed=e1.get('seed', 0), carbon_budget=e1.get('carbon_budget', 1.5),
                        steps_per_hour=e1.get('steps_per_hour', 60), csv_trace=e1.get('csv_trace', None), save_prefix=e1.get('save_prefix', 'exp1'))
    elif args.exp == 2:
        e2 = cfg.get('experiment2', {})
        run_experiment2(img_dir=paths['images'], log_dir=paths['logs'], models_dir=paths['models'],
                        hours=e2.get('hours', 8), seed=e2.get('seed', 1), steps_per_hour=e2.get('steps_per_hour', 60),
                        csv_trace=e2.get('csv_trace', None), save_prefix=e2.get('save_prefix', 'exp2'))
    elif args.exp == 3:
        e3 = cfg.get('experiment3', {})
        run_experiment3(img_dir=paths['images'], log_dir=paths['logs'], models_dir=paths['models'],
                        hours=e3.get('hours', 24), seed=e3.get('seed', 2), slots=e3.get('slots', 2),
                        csv_trace=e3.get('csv_trace', None), save_prefix=e3.get('save_prefix', 'exp3'))
    else:
        print('[Main] No experiment selected. Use --quick_test or --exp {1,2,3}.')


if __name__ == '__main__':
    main()
