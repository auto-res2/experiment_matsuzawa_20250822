import os
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import yaml

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'config')

os.makedirs(DATA_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


class ToyMultiModalSample:
    def __init__(self, text_len, img_len, pattern, label, difficulty, decisive_indices):
        self.text_len = text_len
        self.img_len = img_len
        self.pattern = pattern
        self.label = label
        self.difficulty = difficulty
        self.decisive_indices = decisive_indices


class ToyMultiModalDataset:
    def __init__(self, n_samples=500, n_classes=3, patterns=("aligned", "noisy", "sparse_signal"),
                 text_len_range=(24, 48), img_len_range=(32, 80), seed=42):
        set_seed(seed)
        self.n_classes = n_classes
        self.samples: List[ToyMultiModalSample] = []
        self.patterns = patterns
        for _ in range(n_samples):
            pattern = random.choice(patterns)
            text_len = random.randint(*text_len_range)
            img_len = random.randint(*img_len_range)
            difficulty = random.random()
            n_decisive_txt = max(1, text_len // (8 if pattern != 'sparse_signal' else 20))
            n_decisive_img = max(1, img_len // (8 if pattern != 'sparse_signal' else 20))
            text_decisive = sorted(random.sample(range(text_len), n_decisive_txt))
            img_decisive = sorted(random.sample(range(img_len), n_decisive_img))
            label = random.randint(0, n_classes - 1)
            self.samples.append(ToyMultiModalSample(text_len, img_len, pattern, label, difficulty,
                                                    (text_decisive, img_decisive)))

    def to_records(self) -> List[Dict]:
        recs = []
        for s in self.samples:
            recs.append({
                'text_len': s.text_len,
                'img_len': s.img_len,
                'pattern': s.pattern,
                'label': s.label,
                'difficulty': s.difficulty,
                'decisive_indices': [s.decisive_indices[0], s.decisive_indices[1]]
            })
        return recs


def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path: str = None, force: bool = False):
    if config_path is None:
        config_path = os.path.join(CONFIG_DIR, 'default.yaml')
    cfg = load_config(config_path)

    dataset_json = os.path.join(DATA_DIR, cfg['data']['dataset_json'])
    if os.path.exists(dataset_json) and not force:
        print(f"[Preprocess] Dataset already exists at {dataset_json}. Skipping. Use force=True to regenerate.")
        return

    print("[Preprocess] Generating synthetic multimodal dataset...")
    train_ds = ToyMultiModalDataset(n_samples=cfg['data']['n_train'], n_classes=cfg['data']['n_classes'],
                                    text_len_range=tuple(cfg['data']['text_len_range']), img_len_range=tuple(cfg['data']['img_len_range']), seed=cfg['seed']+1)
    val_ds = ToyMultiModalDataset(n_samples=cfg['data']['n_val'], n_classes=cfg['data']['n_classes'],
                                  text_len_range=tuple(cfg['data']['text_len_range']), img_len_range=tuple(cfg['data']['img_len_range']), seed=cfg['seed']+2)
    test_ds = ToyMultiModalDataset(n_samples=cfg['data']['n_test'], n_classes=cfg['data']['n_classes'],
                                   text_len_range=tuple(cfg['data']['text_len_range']), img_len_range=tuple(cfg['data']['img_len_range']), seed=cfg['seed']+3)

    data = {
        'train': train_ds.to_records(),
        'val': val_ds.to_records(),
        'test': test_ds.to_records(),
        'meta': {
            'n_classes': cfg['data']['n_classes'],
            'patterns': ["aligned", "noisy", "sparse_signal"]
        }
    }

    with open(dataset_json, 'w') as f:
        json.dump(data, f)
    print(f"[Preprocess] Saved dataset to {dataset_json}")


if __name__ == '__main__':
    cfg_path = os.path.join(CONFIG_DIR, 'default.yaml')
    main(cfg_path, force=False)
