import os
import numpy as np
from typing import Dict, Any

from src.train import generate_synth_ex1, generate_regime_carousel_ex3

try:
    import pandas as pd  # optional
except Exception:
    pd = None


def load_electricity_series(csv_path: str = None) -> np.ndarray:
    if csv_path is not None and os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if 'datetime' in df.columns and 'load' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime').asfreq('H')
                df['load'] = df['load'].interpolate('time').ffill().bfill()
                y = df['load'].values.astype(float)
                return y
        except Exception:
            pass
    # Fallback synthetic seasonal weekly data
    T = 35_000
    rng = np.random.default_rng(202)
    base = 100 + 10 * np.sin(2 * np.pi * np.arange(T) / 24.0) + 15 * np.sin(2 * np.pi * np.arange(T) / 168.0)
    drift = np.linspace(0, 20, T)
    noise = rng.normal(0, 5.0, size=T)
    y = base + drift + noise
    return y.astype(float)


def preprocess(config: Dict[str, Any], data_dir: str) -> Dict[str, Any]:
    os.makedirs(data_dir, exist_ok=True)
    exp = str(config.get('experiment', 'test')).lower()
    out = {}

    if exp == 'exp1':
        T = int(config.get('T', 50000))
        seed = int(config.get('seed', 0))
        y, S = generate_synth_ex1(T=T, seed=seed)
        path = os.path.join(data_dir, f'exp1_T{T}_seed{seed}.npz')
        np.savez_compressed(path, y=y, T=T, seed=seed)
        out['data_path'] = path
        out['T'] = T
        out['seed'] = seed
    elif exp == 'exp2':
        csv_path = config.get('electricity_csv_path', None)
        y = load_electricity_series(csv_path)
        path = os.path.join(data_dir, 'exp2_electricity.npz')
        np.savez_compressed(path, y=y)
        out['data_path'] = path
        out['T'] = len(y)
    elif exp == 'exp3':
        T = int(config.get('T', 60000))
        seed = int(config.get('seed', 7))
        y, segments, W_star = generate_regime_carousel_ex3(T=T, seed=seed)
        path = os.path.join(data_dir, f'exp3_T{T}_seed{seed}.npz')
        np.savez_compressed(path, y=y, segments=np.array(segments, dtype=object), W_star=np.array(W_star, dtype=object))
        out['data_path'] = path
        out['T'] = T
        out['seed'] = seed
    else:  # test (quick)
        T = int(config.get('T', 2000))
        seed = int(config.get('seed', 0))
        y, _ = generate_synth_ex1(T=T, seed=seed)
        path = os.path.join(data_dir, f'test_T{T}_seed{seed}.npz')
        np.savez_compressed(path, y=y, T=T, seed=seed)
        out['data_path'] = path
        out['T'] = T
        out['seed'] = seed

    return out
