
import os
import sys
import time
import json
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.preprocess import set_seed, get_device
from src.evaluate import (
    run_synthetic_sde_experiment,
    run_greeks_experiment, 
    run_robustness_experiment,
    create_experiment_plots
)


def print_system_info():
    """Print system and environment information"""
    print("=" * 60)
    print("HiLo-STeP++: High-dimensional Low-rank Stochastic Taylor Probing")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Device: {get_device()}")
    print("=" * 60)


def run_quick_test() -> bool:
    """Run a quick functionality test to ensure everything works"""
    print("\n🧪 Running quick functionality test...")
    
    try:
        set_seed(42)
        device = get_device()
        
        x = torch.randn(5, device=device)
        y = torch.sin(x)
        assert y.shape == x.shape
        
        from torch.autograd.functional import jvp
        def f(x):
            return torch.sin(x).sum()
        v = torch.ones_like(x)
        val, grad = jvp(f, (x,), (v,))
        assert val.numel() == 1
        
        print("✅ Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def run_all_experiments(device=None) -> Dict[str, Any]:
    """Run all three main experiments"""
    if device is None:
        device = get_device()
    
    print("\n🚀 Starting experimental pipeline...")
    results = {}
    
    print("\n" + "="*50)
    print("EXPERIMENT 1: Synthetic SDEs with Bracket Sparsity")
    print("="*50)
    start_time = time.time()
    
    try:
        results_synthetic = run_synthetic_sde_experiment(d=20, device=device)
        results["synthetic_sde"] = results_synthetic
        print(f"✅ Experiment 1 completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"❌ Experiment 1 failed: {e}")
        results["synthetic_sde"] = {"error": str(e)}
    
    print("\n" + "="*50)
    print("EXPERIMENT 2: High-dimensional Greeks under Local Volatility")
    print("="*50)
    start_time = time.time()
    
    try:
        results_greeks = run_greeks_experiment(d=100, rank=5, device=device)
        results["greeks"] = results_greeks
        print(f"✅ Experiment 2 completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"❌ Experiment 2 failed: {e}")
        results["greeks"] = {"error": str(e)}
    
    print("\n" + "="*50)
    print("EXPERIMENT 3: Robustness under Non-smooth Drifts")
    print("="*50)
    start_time = time.time()
    
    try:
        results_robustness = run_robustness_experiment(d=10, device=device)
        results["robustness"] = results_robustness
        print(f"✅ Experiment 3 completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"❌ Experiment 3 failed: {e}")
        results["robustness"] = {"error": str(e)}
    
    return results


def save_results(results: Dict[str, Any], save_dir: str):
    """Save experimental results to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_tensors(results)
    
    results_serializable["metadata"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device())
    }
    
    results_file = os.path.join(save_dir, "experimental_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"📊 Results saved to {results_file}")


def create_summary_report(results: Dict[str, Any]):
    """Create a summary report of experimental results"""
    print("\n" + "="*60)
    print("EXPERIMENTAL SUMMARY REPORT")
    print("="*60)
    
    if "synthetic_sde" in results and "error" not in results["synthetic_sde"]:
        print("\n📈 Synthetic SDE Experiments:")
        for case_name, case_results in results["synthetic_sde"].items():
            if isinstance(case_results, dict) and "variance_reduction" in case_results:
                vr = case_results["variance_reduction"]
                sparsity = case_results["sparsity"]["precision"]
                print(f"  • {case_name}: VR={vr:.2f}×, Sparsity={sparsity:.3f}")
    
    if "greeks" in results and "error" not in results["greeks"]:
        print("\n💰 Greeks Experiment:")
        greeks_results = results["greeks"]
        print(f"  • Rank estimation: {greeks_results['estimated_rank']}/{greeks_results['true_rank']}")
        print(f"  • Variance reduction: {greeks_results['variance_reduction']:.2f}×")
        print(f"  • Delta error: {greeks_results['delta_error']:.6f}")
    
    if "robustness" in results and "error" not in results["robustness"]:
        print("\n🛡️ Robustness Experiment:")
        rob_results = results["robustness"]
        original_var = rob_results[0.0]["variance"]
        best_eps = min(rob_results.keys(), key=lambda eps: rob_results[eps]["variance"] if eps > 0 else float('inf'))
        if best_eps > 0:
            best_var = rob_results[best_eps]["variance"]
            improvement = original_var / best_var
            print(f"  • Best mollification: ε={best_eps}")
            print(f"  • Variance improvement: {improvement:.2f}×")
    
    errors = []
    for exp_name, exp_results in results.items():
        if isinstance(exp_results, dict) and "error" in exp_results:
            errors.append(f"{exp_name}: {exp_results['error']}")
    
    if errors:
        print("\n❌ Errors encountered:")
        for error in errors:
            print(f"  • {error}")
    
    print("\n" + "="*60)


def main():
    """Main experimental pipeline"""
    print_system_info()
    
    set_seed(42)
    device = get_device()
    
    if not run_quick_test():
        print("❌ Quick test failed. Aborting experiments.")
        sys.exit(1)
    
    total_start_time = time.time()
    results = run_all_experiments(device)
    total_time = time.time() - total_start_time
    
    output_dir = ".research/iteration1"
    images_dir = os.path.join(output_dir, "images")
    
    save_results(results, output_dir)
    
    try:
        if ("synthetic_sde" in results and "error" not in results["synthetic_sde"] and
            "greeks" in results and "error" not in results["greeks"] and
            "robustness" in results and "error" not in results["robustness"]):
            
            print("\n📊 Creating publication-quality plots...")
            create_experiment_plots(
                results["synthetic_sde"],
                results["greeks"], 
                results["robustness"],
                images_dir
            )
            print("✅ Plots created successfully!")
        else:
            print("⚠️ Skipping plot creation due to experiment errors")
    except Exception as e:
        print(f"❌ Plot creation failed: {e}")
    
    create_summary_report(results)
    
    print(f"\n🏁 Total experimental time: {total_time:.2f}s")
    print("🎯 HiLo-STeP++ experimental pipeline completed!")
    
    status_file = os.path.join(output_dir, "status.json")
    with open(status_file, 'w') as f:
        json.dump({"status_enum": "stopped", "completion_time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    print(f"📝 Status set to 'stopped' in {status_file}")


if __name__ == "__main__":
    main()
