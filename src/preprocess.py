"""
Preprocessing and Environment Setup for GASP Experiment

This module handles environment setup, device configuration, and
preprocessing tasks required for the GASP latency benchmark.
"""

import torch
import os
import sys
import platform
import subprocess
from pathlib import Path

def setup_experiment_environment():
    """
    Set up the experimental environment and verify system requirements.
    """
    print("Setting up experiment environment...")
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
            
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # For better performance
        
    else:
        print("CUDA not available - running on CPU")
        
        torch.set_num_threads(min(8, os.cpu_count() or 4))  # Limit threads for stability
        print(f"Using {torch.get_num_threads()} CPU threads")
    
    setup_directories()
    
    verify_dependencies()
    
    print("Environment setup complete!\n")

def setup_directories():
    """Create necessary directories for the experiment."""
    base_path = Path(__file__).parent.parent
    
    directories = [
        base_path / ".research" / "iteration1" / "images",
        base_path / "config",
        base_path / "data",
        base_path / "models"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {directory}")

def verify_dependencies():
    """Verify that all required dependencies are available."""
    required_packages = [
        'torch',
        'pandas', 
        'seaborn',
        'matplotlib',
        'tqdm',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} missing")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        raise ImportError(f"Missing required packages: {missing_packages}")
    
    print("✓ All dependencies verified")

def check_gpu_compatibility():
    """
    Check if the current GPU is compatible with the experiment requirements.
    Specifically designed for Tesla T4 compatibility.
    """
    if not torch.cuda.is_available():
        return {
            'compatible': False,
            'reason': 'CUDA not available',
            'recommendations': ['Use CPU mode for testing']
        }
    
    device_props = torch.cuda.get_device_properties(0)
    gpu_memory_gb = device_props.total_memory / 1e9
    gpu_name = device_props.name
    
    print(f"GPU Analysis: {gpu_name}")
    print(f"Memory: {gpu_memory_gb:.1f} GB")
    
    min_memory_gb = 4.0  # Minimum for small models
    recommended_memory_gb = 16.0  # Tesla T4 specification
    
    recommendations = []
    
    if gpu_memory_gb < min_memory_gb:
        return {
            'compatible': False,
            'reason': f'Insufficient GPU memory ({gpu_memory_gb:.1f} GB < {min_memory_gb} GB)',
            'recommendations': ['Use CPU mode', 'Reduce model size']
        }
    
    if gpu_memory_gb < recommended_memory_gb:
        recommendations.extend([
            'Reduce model embedding dimension',
            'Reduce number of layers',
            'Use smaller batch sizes'
        ])
    
    compute_capability = f"{device_props.major}.{device_props.minor}"
    min_compute = 6.0  # Minimum for modern PyTorch features
    
    if device_props.major < 6:
        recommendations.append('GPU compute capability may be too old for optimal performance')
    
    return {
        'compatible': True,
        'gpu_name': gpu_name,
        'memory_gb': gpu_memory_gb,
        'compute_capability': compute_capability,
        'recommendations': recommendations
    }

def optimize_for_device(device_type='auto'):
    """
    Optimize PyTorch settings for the target device.
    
    Args:
        device_type: 'auto', 'cuda', or 'cpu'
    """
    if device_type == 'auto':
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device_type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        torch.cuda.empty_cache()
        
        if torch.cuda.is_bf16_supported():
            print("✓ BFloat16 supported - using for mixed precision")
        elif torch.cuda.is_available():
            print("✓ Float16 supported - using for mixed precision")
        
    else:
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    print(f"✓ Optimized for {device_type.upper()}")
    return device_type

def get_optimal_model_config(device_type='auto'):
    """
    Get optimal model configuration based on available hardware.
    
    Args:
        device_type: Target device type
        
    Returns:
        Dictionary with optimal model parameters
    """
    if device_type == 'auto':
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device_type == 'cuda':
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = gpu_props.total_memory / 1e9
        
        if gpu_memory_gb >= 16:  # Tesla T4 or better
            config = {
                'embed_dim': 1024,
                'num_layers': 16,
                'num_heads': 16,
                'num_runs': 100,
                'warmup_runs': 20
            }
        elif gpu_memory_gb >= 8:  # Mid-range GPU
            config = {
                'embed_dim': 768,
                'num_layers': 12,
                'num_heads': 12,
                'num_runs': 50,
                'warmup_runs': 10
            }
        else:  # Low-memory GPU
            config = {
                'embed_dim': 512,
                'num_layers': 8,
                'num_heads': 8,
                'num_runs': 25,
                'warmup_runs': 5
            }
    else:  # CPU
        config = {
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 4,
            'num_runs': 10,
            'warmup_runs': 2
        }
    
    print(f"Optimal config for {device_type}: {config}")
    return config

def validate_experiment_setup():
    """
    Validate that the experiment setup is correct and ready to run.
    
    Returns:
        Boolean indicating if setup is valid
    """
    try:
        base_path = Path(__file__).parent.parent
        required_dirs = [
            base_path / ".research" / "iteration1" / "images",
            base_path / "src"
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                print(f"✗ Missing directory: {directory}")
                return False
        
        gpu_check = check_gpu_compatibility()
        if not gpu_check['compatible']:
            print(f"⚠ GPU compatibility issue: {gpu_check['reason']}")
            print("Recommendations:", gpu_check['recommendations'])
        
        test_tensor = torch.randn(10, 10)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()
            _ = torch.mm(test_tensor, test_tensor)
            torch.cuda.synchronize()
        
        print("✓ Experiment setup validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Experiment setup validation failed: {e}")
        return False

if __name__ == '__main__':
    setup_experiment_environment()
    
    if validate_experiment_setup():
        print("\n✓ Ready to run GASP experiment!")
    else:
        print("\n✗ Setup issues detected. Please resolve before running experiment.")
        sys.exit(1)
