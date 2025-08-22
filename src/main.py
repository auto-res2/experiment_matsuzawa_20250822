import os
import sys
import yaml

# Make local src directory importable when running as `python src/main.py`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import preprocess as preprocess_mod
import train as train_mod
import evaluate as evaluate_mod


def main(config_path: str = 'config/config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    images_dir = cfg['paths'].get('images_dir', '.research/iteration1/images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)

    print('=== Stage 1: Preprocess (geometry / BT) ===')
    preprocess_mod.run(config_path)

    print('\n=== Stage 2: Train models (baseline + TaReS) ===')
    train_mod.run(config_path)

    print('\n=== Stage 3: Evaluate (certification, spoof-resistance, transitivity) ===')
    evaluate_mod.run(config_path)

    print('\nPipeline complete. All figures saved under:', images_dir)


if __name__ == '__main__':
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config/config.yaml'
    main(cfg_path)
