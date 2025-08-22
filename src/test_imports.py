#!/usr/bin/env python

"""
Test script to verify all imports work correctly.
"""

import sys
import os
sys.path.append('src')

try:
    from models import create_model
    print("✓ models.create_model imported successfully")
    
    from preprocess import create_datasets, save_sample_images
    print("✓ preprocess functions imported successfully")
    
    from train import train_model
    print("✓ train.train_model imported successfully")
    
    from evaluate import run_evaluation
    print("✓ evaluate.run_evaluation imported successfully")
    
    print("\n✓ All imports successful!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
