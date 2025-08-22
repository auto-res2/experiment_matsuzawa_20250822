# Orchestrator for PACER-FL experiments (src/main.py)
# - Loads YAML config (optional)
# - Runs experiments and saves outputs to the required directories

import os
import argparse
import yaml

from src.preprocess import ensure_dirs
from src.train import (
    run_quick_test,
    experiment1_pacer_vs_baselines,
    experiment2_comm_efficiency,
    experiment3_auditability,
)


def load_config(path: str):
    if path is None or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="PACER-FL Experiments")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to YAML config file')
    parser.add_argument('--exp', type=int, default=0, help='0=quick-test, 1=exp1, 2=exp2, 3=exp3')
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config(args.config)

    # Apply defaults for required dirs
    cfg.setdefault('images_dir', '.research/iteration1/images')
    cfg.setdefault('models_dir', 'models')

    if args.exp == 0:
        run_quick_test()
        return

    if args.exp == 1:
        res = experiment1_pacer_vs_baselines(cfg)
        print("Exp1 done. Ledger:", res['ledger_path'])
    elif args.exp == 2:
        res = experiment2_comm_efficiency(cfg)
        print("Exp2 done. Last acc:", res['accs_pacer'][-1])
    elif args.exp == 3:
        res = experiment3_auditability(cfg)
        print("Exp3 done. Auditor:", 'PASS' if res['auditor_ok'] else 'FAIL')
    else:
        print("Unknown exp id. Running quick test instead.")
        run_quick_test()


if __name__ == '__main__':
    main()
