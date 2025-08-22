#!/usr/bin/env python3
"""
Simple test to verify all Rev-SS2D modules import and instantiate correctly
"""

import sys
import os
sys.path.append('src')

def test_imports():
    """Test that all modules import correctly"""
    print("Testing imports...")
    
    try:
        from preprocess import SyntheticDataGenerator, DataPreprocessor
        print("✓ preprocess module imported successfully")
        
        from train import (RevSS2DTrainer, BaselineTrainer, RevSS2DModel, 
                          BaselineModel, RevSS2DBlock, SharedStateSS2D)
        print("✓ train module imported successfully")
        
        from evaluate import ModelEvaluator, MetricsTracker
        print("✓ evaluate module imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_instantiation():
    """Test that classes can be instantiated"""
    print("\nTesting class instantiation...")
    
    try:
        from preprocess import SyntheticDataGenerator
        from evaluate import ModelEvaluator
        
        data_gen = SyntheticDataGenerator()
        print("✓ SyntheticDataGenerator instantiated")
        
        evaluator = ModelEvaluator()
        print("✓ ModelEvaluator instantiated")
        
        return True
    except Exception as e:
        print(f"✗ Instantiation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_generation():
    """Test synthetic data generation"""
    print("\nTesting data generation...")
    
    try:
        from preprocess import SyntheticDataGenerator
        
        data_gen = SyntheticDataGenerator()
        
        data = data_gen.generate_classification_data(
            batch_size=2, channels=16, height=64, width=64, num_classes=5
        )
        
        print(f"✓ Classification data generated: {data['images'].shape}")
        print(f"✓ Labels shape: {data['labels'].shape}")
        
        seg_data = data_gen.generate_segmentation_data(
            batch_size=2, channels=16, height=64, width=64, num_classes=5
        )
        
        print(f"✓ Segmentation data generated: {seg_data['images'].shape}")
        print(f"✓ Masks shape: {seg_data['masks'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Data generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Rev-SS2D Module Test")
    print("=" * 30)
    
    success = True
    
    success &= test_imports()
    success &= test_instantiation() 
    success &= test_data_generation()
    
    print("\n" + "=" * 30)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    return success

if __name__ == "__main__":
    main()
