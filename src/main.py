#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main runner for OP-S&V experiments (Iteration 1).

- Loads configuration from config/config.yaml
- Runs preprocessing to build synthetic datasets
- Trains using OP-S&V selection and baselines
- Evaluates and saves paper-ready PDF figures into .research/iteration1/images

To run:
  python -m src.main
"""

import os
import sys
import json
import yaml

from . import preprocess as prep
from . import train as trainer


def ensure_dirs():
    os.makedirs('.research/iteration1/images', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)


def load_config(cfg_path: str = 'config/config.yaml') -> dict:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found at {cfg_path}. Please create it or use the provided default.")
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    ensure_dirs()
    cfg = load_config()
    print("Loaded config:\n", json.dumps(cfg, indent=2))

    # Preprocess / build datasets
    datasets = prep.preprocess(cfg)

    # Run experiments
    results = trainer.run_experiments(datasets, cfg)

    # Summaries on stdout
    print("\n=== Experiment Summaries ===")
    if 'exp1' in results:
        exp1 = results['exp1']
        print(f"[Exp1] Budgets: {exp1['budgets']}")
        print(f"[Exp1] Nestedness: {exp1['nestedness']:.3f}")
        print(f"[Exp1] Selection time (s): {exp1['selection_time_sec']:.3f}")
        print("[Exp1] Accuracy per method:")
        for m, vals in exp1['methods'].items():
            print(f"  {m:8s}: ", [f"{a:.3f}" for a in vals['acc']])
    if 'exp2' in results:
        exp2 = results['exp2']
        print(f"[Exp2] Budgets: {exp2['budgets']}")
        print("[Exp2] Overall vs worst-group accuracies (pairs) per method:")
        for m, vals in exp2['methods'].items():
            pairs = list(zip(vals['acc'], vals['worst_group_acc']))
            print(f"  {m:8s}: ", [f"({a:.3f},{w:.3f})" for a, w in pairs])

    # Confirm PDFs saved
    images_dir = cfg['experiment'].get('images_dir', '.research/iteration1/images')
    print("\nPDFs saved (for paper-ready figures):")
    for f in sorted(os.listdir(images_dir)):
        if f.lower().endswith('.pdf'):
            print("  ", os.path.join(images_dir, f))


if __name__ == "__main__":
    main()
