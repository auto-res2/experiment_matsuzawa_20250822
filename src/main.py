import os
import yaml

from src.preprocess import main as preprocess_main
from src.train import train_model
from src.evaluate import run_synthetic_experiments_from_config, evaluate_trained_model


def ensure_dirs(cfg_path: str = "config/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Create required directories
    os.makedirs(".research/iteration1/images", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def run_all(config_path: str = "config/config.yaml"):
    ensure_dirs(config_path)
    print("========== Stage 1: Preprocess ==========")
    preprocess_main(config_path)

    print("\n========== Stage 2: Train ==========")
    train_model(config_path)

    print("\n========== Stage 3: Evaluate (Synthetic Experiments) ==========")
    run_synthetic_experiments_from_config(config_path)

    print("\n========== Stage 4: Evaluate (Trained Model with CS-SHAP) ==========")
    evaluate_trained_model(config_path)

    print("\n========== Pipeline Complete ==========")


if __name__ == "__main__":
    run_all()
