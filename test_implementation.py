#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing imports...")
    from src.preprocess import set_seed, get_device
    print("✅ Preprocessing module import successful")
    
    from src.train import ItoSDE, CriticNet
    print("✅ Training module import successful")
    
    from src.evaluate import bracket_sparsity_screen
    print("✅ Evaluation module import successful")
    
    from src.main import run_quick_test
    print("✅ Main module import successful")
    
    print("\nRunning quick functionality test...")
    success = run_quick_test()
    
    if success:
        print("🎉 All tests passed! Implementation is ready.")
        sys.exit(0)
    else:
        print("❌ Quick test failed!")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Import or test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
