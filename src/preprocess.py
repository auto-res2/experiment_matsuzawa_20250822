# -*- coding: utf-8 -*-
"""
preprocess.py

Creates synthetic datasets (FragBench-like) for tasks used in experiments and saves them to data/.
"""
import os
import json
from typing import Dict, Any, Tuple, List

from train import make_synthetic_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_DEFAULT = os.path.join(BASE_DIR, 'data')


def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def preprocess_all(seed: int = 0, data_dir: str = DATA_DIR_DEFAULT) -> Dict[str, str]:
    tasks = {
        'gsm8k': (120, 240),
        'mmlu': (100, 200),
        'bbh': (100, 80),
    }
    paths: Dict[str, str] = {}
    for task, (n_dev, n_test) in tasks.items():
        dev, test = make_synthetic_dataset(task, n_dev=n_dev, n_test=n_test, seed=seed)
        dev_path = os.path.join(data_dir, f'{task}_dev.jsonl')
        test_path = os.path.join(data_dir, f'{task}_test.jsonl')
        _write_jsonl(dev_path, dev)
        _write_jsonl(test_path, test)
        paths[f'{task}_dev'] = dev_path
        paths[f'{task}_test'] = test_path
        print(f"Wrote {task} dev/test to {dev_path}, {test_path}")
    return paths
