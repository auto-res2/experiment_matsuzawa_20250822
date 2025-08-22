"""
Configuration for SEEDS experiments
"""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BetaParams:
    beta0: float = 0.1
    beta1: float = 20.0
    gamma: float = 2.0

@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cuda"
    
    K: int = 16  # Number of discrete states
    
    beta_params: BetaParams = field(default_factory=BetaParams)
    
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    
    delta: float = 0.01  # Target violation rate for calibration
    kappa_init: float = 2.0  # Initial kappa multiplier
    p_corr: float = 0.8  # Correction probability in budgeted mode
    
    image_size: int = 8  # Small for T4 GPU
    seq_length: int = 32
    n_samples: int = 1000
    n_test_samples: int = 100
    
    results_dir: str = ".research/iteration1/images"
    models_dir: str = "models/checkpoints"
    data_dir: str = "data"

config = ExperimentConfig()
