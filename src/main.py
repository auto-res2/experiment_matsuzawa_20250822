import os
import yaml

from train import ensure_dir


def run_orchestrator(config_path: str = 'config/config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Ensure directories
    ensure_dir(cfg.get('models_dir', 'models'))
    ensure_dir(cfg.get('data_dir', 'data'))
    ensure_dir(cfg.get('output_image_dir', '.research/iteration1/images'))

    # 1) Pretrain vision model
    print('[Main] Pretraining vision toy model...')
    os.system(f"python -m src.train --config {config_path} --task vision")

    # 2) Offline preprocess (learn P, fit S)
    print('[Main] Offline precompute P and S (vision)...')
    os.system(f"python -m src.preprocess --config {config_path}")

    # 3) Evaluate & adapt across experiments
    print('[Main] Running evaluation & adaptation experiments...')
    os.system(f"python -m src.evaluate --config {config_path}")

    print('[Main] Done. Figures saved under', cfg.get('output_image_dir', '.research/iteration1/images'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SketchBank Orchestrator')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML')
    args = parser.parse_args()
    run_orchestrator(args.config)
