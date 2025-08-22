# -*- coding: utf-8 -*-
"""
evaluate.py

Evaluation utilities for SURGE-Prompt experiments.
Loads trained artifacts if needed, evaluates prompts across settings, and writes summary JSON.
"""
import os
import json
from typing import Dict, Any, List, Tuple

import numpy as np

from train import (
    TargetBlackBox,
    eval_on_test_settings,
    to_pir_from_seed,
    SURGE,
    FragmentBank,
    Validator,
    make_synthetic_dataset,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR_DEFAULT = os.path.join(BASE_DIR, '.research', 'iteration1', 'images')
MODELS_DIR_DEFAULT = os.path.join(BASE_DIR, 'models')
DATA_DIR_DEFAULT = os.path.join(BASE_DIR, 'data')


def evaluate_saved_surrogate(seed: int = 0, images_dir: str = IMAGES_DIR_DEFAULT, models_dir: str = MODELS_DIR_DEFAULT) -> Dict[str, Any]:
    target = TargetBlackBox(seed)
    surge = SURGE(target, seed)
    model_path = os.path.join(models_dir, 'surge_surrogate_exp1.pt')
    if os.path.exists(model_path):
        surge.load(models_dir=models_dir, name='surge_surrogate_exp1.pt')
    else:
        print("Warning: No saved surrogate from Exp1. Proceeding with fresh model state.")

    bank = surge.bank
    val = surge.val
    dev, test = make_synthetic_dataset('gsm8k', n_dev=60, n_test=120, seed=seed)

    seed_prompt = 'You are an expert math tutor. Think systematically and respond in JSON.'
    pir0 = to_pir_from_seed(seed_prompt, bank)
    pir0 = val.repair(pir0)

    # Simple evaluation of the seed PIR to show pipeline behavior
    results = eval_on_test_settings(target, pir0, test, router=None)

    out_json = os.path.join(models_dir, 'evaluation_seed_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {out_json}")
    return results
