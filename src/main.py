#!/usr/bin/env python3
"""
AST-SelfCheck: Calibrated, test-optional, per-line P(error) for self-debugging code LLMs

Main experimental script that orchestrates the full pipeline from preprocessing to evaluation.
"""

import os
import sys
import time
import json
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_data, set_seed
from train import train_models
from evaluate import evaluate_models


def update_status(status: str):
    """Update experiment status."""
    status_file = ".research/status.json"
    status_data = {"status_enum": status, "timestamp": time.time()}
    
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    
    with open(status_file, "w") as f:
        json.dump(status_data, f, indent=2)
    
    print(f"Status updated to: {status}")


def run_experiment() -> Dict[str, Any]:
    """Run the complete AST-SelfCheck experiment."""
    print("=" * 60)
    print("AST-SelfCheck: Calibrated Error Detection for Code LLMs")
    print("=" * 60)
    
    set_seed(42)
    
    update_status("running")
    
    try:
        print("\n" + "=" * 40)
        print("STEP 1: DATA PREPROCESSING")
        print("=" * 40)
        
        codes, labels, features, fault_lines = preprocess_data()
        print(f"✓ Generated {len(codes)} code samples")
        print(f"✓ Extracted AST features: {len(features[0])} dimensions")
        print(f"✓ Buggy samples: {sum(labels)}/{len(labels)} ({100*sum(labels)/len(labels):.1f}%)")
        
        print("\n" + "=" * 40)
        print("STEP 2: MODEL TRAINING")
        print("=" * 40)
        
        model_wrapper, conformal_predictor, training_info = train_models()
        print(f"✓ Trained MLP classifier with temperature scaling")
        print(f"✓ Trained conformal predictor for uncertainty quantification")
        print(f"✓ Test accuracy: {training_info['test_accuracy']:.3f}")
        
        print("\n" + "=" * 40)
        print("STEP 3: MODEL EVALUATION")
        print("=" * 40)
        
        evaluation_results = evaluate_models()
        print(f"✓ AUC: {evaluation_results['auc']:.3f}")
        print(f"✓ Average Precision: {evaluation_results['average_precision']:.3f}")
        print(f"✓ Calibration ECE: {evaluation_results['calibration']['ECE']:.3f}")
        print(f"✓ Generated evaluation plots in PDF format")
        
        print("\n" + "=" * 40)
        print("EXPERIMENT SUMMARY")
        print("=" * 40)
        
        summary = {
            "experiment_name": "AST-SelfCheck",
            "total_samples": len(codes),
            "feature_dimensions": len(features[0]),
            "test_accuracy": training_info['test_accuracy'],
            "auc_score": evaluation_results['auc'],
            "average_precision": evaluation_results['average_precision'],
            "calibration_ece": evaluation_results['calibration']['ECE'],
            "calibration_mce": evaluation_results['calibration']['MCE'],
            "brier_score": evaluation_results['calibration']['Brier'],
            "function_evaluations": len(evaluation_results.get('function_results', [])),
            "plots_generated": [
                "roc_curve.pdf",
                "precision_recall_curve.pdf", 
                "calibration_plot.pdf",
                "confusion_matrix.pdf"
            ]
        }
        
        summary_file = ".research/iteration1/experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print("Key Results:")
        print(f"  • Error Detection AUC: {summary['auc_score']:.3f}")
        print(f"  • Average Precision: {summary['average_precision']:.3f}")
        print(f"  • Calibration ECE: {summary['calibration_ece']:.3f}")
        print(f"  • Test Accuracy: {summary['test_accuracy']:.3f}")
        print(f"  • Plots saved: {len(summary['plots_generated'])} PDF files")
        
        print("\nMethodology Highlights:")
        print("  • Conservative SpecMining DSL for test generation")
        print("  • Regenerate-and-Compare at AST level (R&C-AST)")
        print("  • Hybrid verification (dynamic + static + symbolic)")
        print("  • Spectrum-Based Fault Localization (SBFL)")
        print("  • Meta-calibrator with conformal prediction")
        print("  • Hierarchical P(error) with confidence intervals")
        
        update_status("stopped")
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return summary
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        update_status("failed")
        
        raise e


def test_quick():
    """Quick functionality test."""
    print("Running quick functionality test...")
    
    try:
        from preprocess import generate_sample_functions, ast_feature_counts
        from train import MLPClassifier, TemperatureScaler
        from evaluate import evaluate_calibration
        print("✓ All imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        sample_code = '''def test_func(x):
    return x + 1'''
        features = ast_feature_counts(sample_code)
        assert isinstance(features, dict)
        print("✓ AST feature extraction works")
        
        model = MLPClassifier(10)
        assert model is not None
        print("✓ Model creation works")
        
        y_true = [0, 1, 0, 1]
        y_prob = [0.1, 0.9, 0.2, 0.8]
        cal_metrics = evaluate_calibration(np.array(y_true), np.array(y_prob))
        assert 'ECE' in cal_metrics
        print("✓ Calibration metrics work")
        
        print("✓ Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AST-SelfCheck Experiment")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run quick functionality test only")
    
    args = parser.parse_args()
    
    if args.quick_test:
        import numpy as np
        success = test_quick()
        sys.exit(0 if success else 1)
    else:
        run_experiment()
