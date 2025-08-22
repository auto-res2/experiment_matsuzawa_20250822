import os
import yaml
import numpy as np

# Non-interactive backend for reproducible image generation (if needed)
import matplotlib
matplotlib.use("Agg")


def set_seed(seed: int = 42):
    np.random.seed(seed)


def generate_synthetic_tabular(n_train=2000, n_val=200, n_test=200, d=32, noise_std=0.1, seed=42):
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    X = rng.normal(0, 1, size=(n, d)).astype(np.float64)

    # Nonlinear target: linear + pairwise + sinusoidal ridge
    a = rng.normal(0, 1, size=d)
    a *= rng.choice([-1, 1], size=d) * rng.uniform(0.5, 2.0, size=d)

    # Sparse pairwise interactions
    s_pairs = max(1, d // 4)
    pairs = set()
    while len(pairs) < s_pairs:
        i, j = sorted(rng.choice(d, size=2, replace=False).tolist())
        if i != j:
            pairs.add((i, j))
    b = {p: rng.normal(0, 0.5) for p in pairs}

    c = rng.normal(0, 1, size=d)
    y = X @ a + np.sin(X @ c)
    for (i, j), wij in b.items():
        y += wij * X[:, i] * X[:, j]
    y += rng.normal(0, noise_std, size=n)

    # Split
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def main(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    data_dir = cfg.get("data", {}).get("dir", "data")
    os.makedirs(data_dir, exist_ok=True)

    n_train = int(cfg.get("data", {}).get("n_train", 2000))
    n_val = int(cfg.get("data", {}).get("n_val", 200))
    n_test = int(cfg.get("data", {}).get("n_test", 200))
    d = int(cfg.get("data", {}).get("d", 32))
    noise_std = float(cfg.get("data", {}).get("noise_std", 0.1))
    data_name = cfg.get("data", {}).get("name", "synthetic_tabular")

    dataset = generate_synthetic_tabular(n_train, n_val, n_test, d, noise_std, seed)

    out_path = os.path.join(data_dir, f"{data_name}.npz")
    np.savez_compressed(out_path, **dataset)
    print(f"[Preprocess] Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
