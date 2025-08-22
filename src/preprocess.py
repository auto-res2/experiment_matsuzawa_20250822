"""
QELO Preprocessing Module
Handles data loading, tokenization, and calibration data preparation.
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import random


def set_seeds(seed: int = 0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CalibrationDataLoader:
    """Handles loading and preparation of calibration data for QELO."""
    
    def __init__(self, 
                 dataset_name: str = "wikitext",
                 dataset_config: str = "wikitext-103-v1",
                 tokenizer_name: str = "EleutherAI/pythia-410m",
                 max_length: int = 2048,
                 num_samples: int = 512):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.num_samples = num_samples
        self.tokenizer = None
        self.calibration_data = None
        
    def load_tokenizer(self):
        """Load the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def load_calibration_data(self) -> List[torch.Tensor]:
        """Load and tokenize calibration data."""
        if self.tokenizer is None:
            self.load_tokenizer()
            
        if self.dataset_name == "wikitext":
            dataset = load_dataset(self.dataset_name, self.dataset_config, split="train")
        else:
            dataset = load_dataset(self.dataset_name, split="train")
            
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 50)
        
        calibration_sequences = []
        for i, example in enumerate(dataset):
            if len(calibration_sequences) >= self.num_samples:
                break
                
            text = example["text"].strip()
            if len(text) < 50:  # Skip very short texts
                continue
                
            if self.tokenizer is not None:
                tokens = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding=False,
                    return_tensors="pt"
                )
                
                input_ids = tokens["input_ids"].squeeze(0)
                if len(input_ids) >= 128:  # Minimum sequence length
                    calibration_sequences.append(input_ids)
                
        print(f"Loaded {len(calibration_sequences)} calibration sequences")
        self.calibration_data = calibration_sequences
        return calibration_sequences
    
    def get_evaluation_data(self, dataset_name: str = "wikitext", 
                           dataset_config: str = "wikitext-2-v1",
                           max_samples: int = 1000) -> List[torch.Tensor]:
        """Load evaluation data for perplexity calculation."""
        if self.tokenizer is None:
            self.load_tokenizer()
            
        if dataset_name == "wikitext":
            dataset = load_dataset(dataset_name, dataset_config, split="test")
        else:
            dataset = load_dataset(dataset_name, split="test")
            
        eval_sequences = []
        for i, example in enumerate(dataset):
            if len(eval_sequences) >= max_samples:
                break
                
            text = example["text"].strip()
            if len(text) < 50:
                continue
                
            if self.tokenizer is not None:
                tokens = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding=False,
                    return_tensors="pt"
                )
                
                input_ids = tokens["input_ids"].squeeze(0)
                if len(input_ids) >= 32:
                    eval_sequences.append(input_ids)
                
        print(f"Loaded {len(eval_sequences)} evaluation sequences")
        return eval_sequences


def create_synthetic_data(num_samples: int = 1000, 
                         d_in: int = 768, 
                         d_out: int = 768,
                         patterns: List[str] = ["gauss_iso", "gauss_aniso"],
                         seed: int = 0) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Create synthetic data for testing QELO components."""
    set_seeds(seed)
    
    synthetic_data = {}
    
    for pattern in patterns:
        if pattern == "gauss_iso":
            X = torch.randn(num_samples, d_in)
        elif pattern == "gauss_aniso":
            X = torch.randn(num_samples, d_in)
            scale = torch.linspace(0.5, 3.0, d_in)
            X = X * scale
        elif pattern == "laplace":
            u = torch.rand(num_samples, d_in) - 0.5
            X = -torch.sign(u) * torch.log1p(-2 * u.abs())
        elif pattern == "mixture":
            mix = torch.rand(num_samples, 1)
            comp1 = torch.randn(num_samples, d_in)
            comp2 = torch.randn(num_samples, d_in) + 2.5
            X = (mix > 0.7).float() * comp2 + (mix <= 0.7).float() * comp1
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
            
        W = torch.randn(d_out, d_in) * 0.1
        Y = X @ W.t()
        
        synthetic_data[pattern] = (X.to(torch.float32), Y.to(torch.float32))
        
    return synthetic_data


def prepare_model_data(model_name: str = "EleutherAI/pythia-410m") -> Dict:
    """Prepare model configuration and basic info."""
    return {
        "model_name": model_name,
        "tokenizer_name": model_name,
        "supported_models": [
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1.4b",
            "facebook/opt-1.3b"
        ]
    }


if __name__ == "__main__":
    set_seeds(42)
    
    print("Testing synthetic data generation...")
    synthetic_data = create_synthetic_data(
        num_samples=100,
        d_in=64,
        d_out=64,
        patterns=["gauss_iso", "gauss_aniso"]
    )
    
    for pattern, (X, Y) in synthetic_data.items():
        print(f"{pattern}: X shape {X.shape}, Y shape {Y.shape}")
        print(f"  X stats: mean={X.mean():.3f}, std={X.std():.3f}")
        print(f"  Y stats: mean={Y.mean():.3f}, std={Y.std():.3f}")
    
    print("\nTesting calibration data loader...")
    try:
        loader = CalibrationDataLoader(num_samples=10)
        loader.load_tokenizer()
        print(f"Tokenizer loaded: {loader.tokenizer_name}")
        print("Preprocessing module test completed successfully!")
    except Exception as e:
        print(f"Error in calibration data loading: {e}")
        print("This is expected if datasets are not available - will work in main pipeline")
