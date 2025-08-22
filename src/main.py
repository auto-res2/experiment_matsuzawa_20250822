"""
BEMeGA Main Experiment Script
Orchestrates the complete experimental pipeline for few-shot learning robustness evaluation
"""

import os
import sys
import json
import time
from typing import Dict, Any
import torch
import numpy as np
import matplotlib.pyplot as plt

from preprocess import set_seed, SyntheticEpisodeGenerator
from train import BEMeGAAdapter, build_random_dictionary
from evaluate import (
    run_k_mismatch_experiment, 
    run_anisotropy_experiment, 
    run_domain_shift_experiment,
    plot_results,
    plot_diagnostics
)


def print_system_info():
    """Print system and GPU information"""
    print("=" * 60)
    print("BEMeGA Experiment System Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("CUDA not available - using CPU")
    
    print("=" * 60)


def create_summary_plot(all_results: Dict, save_dir: str):
    """Create a comprehensive summary plot of all experiments"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    k_results = all_results['k_mismatch']
    k_values = sorted(k_results.keys())
    k_bemega = [k_results[k]['bemega']['mean_accuracy'] for k in k_values]
    k_protonet = [k_results[k]['protonet']['mean_accuracy'] for k in k_values]
    
    axes[0, 0].plot(k_values, k_bemega, 'o-', label='BEMeGA', linewidth=2, markersize=6)
    axes[0, 0].plot(k_values, k_protonet, 's-', label='ProtoNet', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Test k-shot')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('K-Mismatch Robustness')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    aniso_results = all_results['anisotropy']
    aniso_factors = sorted(aniso_results.keys())
    aniso_bemega = [aniso_results[f]['bemega']['mean_accuracy'] for f in aniso_factors]
    aniso_protonet = [aniso_results[f]['protonet']['mean_accuracy'] for f in aniso_factors]
    
    axes[0, 1].plot(aniso_factors, aniso_bemega, 'o-', label='BEMeGA', linewidth=2, markersize=6)
    axes[0, 1].plot(aniso_factors, aniso_protonet, 's-', label='ProtoNet', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Anisotropy Factor')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Anisotropy Robustness')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    domain_results = all_results['domain_shift']
    shift_factors = sorted(domain_results.keys())
    domain_bemega = [domain_results[f]['bemega']['mean_accuracy'] for f in shift_factors]
    domain_protonet = [domain_results[f]['protonet']['mean_accuracy'] for f in shift_factors]
    domain_trans_rates = [domain_results[f]['bemega']['transductive_rate'] for f in shift_factors]
    
    axes[1, 0].plot(shift_factors, domain_bemega, 'o-', label='BEMeGA', linewidth=2, markersize=6)
    axes[1, 0].plot(shift_factors, domain_protonet, 's-', label='ProtoNet', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Domain Shift Factor')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Domain Shift Robustness')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(shift_factors, domain_trans_rates, 'ro-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Domain Shift Factor')
    axes[1, 1].set_ylabel('Transductive Call Rate')
    axes[1, 1].set_title('Adaptive Transduction Usage')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bemega_comprehensive_results.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive summary plot saved to: {os.path.join(save_dir, 'bemega_comprehensive_results.pdf')}")


def print_experiment_summary(all_results: Dict):
    """Print a detailed summary of experimental results"""
    print("\n" + "=" * 80)
    print("BEMEGA EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n1. K-MISMATCH ROBUSTNESS EXPERIMENT")
    print("-" * 50)
    k_results = all_results['k_mismatch']
    for k in sorted(k_results.keys()):
        bemega_acc = k_results[k]['bemega']['mean_accuracy']
        protonet_acc = k_results[k]['protonet']['mean_accuracy']
        improvement = (bemega_acc - protonet_acc) * 100
        print(f"k={k:2d}: BEMeGA={bemega_acc:.3f}, ProtoNet={protonet_acc:.3f}, "
              f"Improvement={improvement:+.1f}%")
    
    print("\n2. ANISOTROPY ROBUSTNESS EXPERIMENT")
    print("-" * 50)
    aniso_results = all_results['anisotropy']
    for factor in sorted(aniso_results.keys()):
        bemega_acc = aniso_results[factor]['bemega']['mean_accuracy']
        protonet_acc = aniso_results[factor]['protonet']['mean_accuracy']
        improvement = (bemega_acc - protonet_acc) * 100
        print(f"Factor={factor:.1f}: BEMeGA={bemega_acc:.3f}, ProtoNet={protonet_acc:.3f}, "
              f"Improvement={improvement:+.1f}%")
    
    print("\n3. DOMAIN SHIFT ROBUSTNESS EXPERIMENT")
    print("-" * 50)
    domain_results = all_results['domain_shift']
    for factor in sorted(domain_results.keys()):
        bemega_acc = domain_results[factor]['bemega']['mean_accuracy']
        protonet_acc = domain_results[factor]['protonet']['mean_accuracy']
        trans_rate = domain_results[factor]['bemega']['transductive_rate']
        improvement = (bemega_acc - protonet_acc) * 100
        print(f"Shift={factor:.1f}: BEMeGA={bemega_acc:.3f}, ProtoNet={protonet_acc:.3f}, "
              f"Improvement={improvement:+.1f}%, Trans.Rate={trans_rate:.2f}")
    
    print("\n" + "=" * 80)


def quick_test_run(device: str = "cpu") -> bool:
    """Run a quick test to verify all components work"""
    print("\n" + "=" * 60)
    print("RUNNING QUICK FUNCTIONALITY TEST")
    print("=" * 60)
    
    try:
        print("Testing preprocessing module...")
        set_seed(42)
        generator = SyntheticEpisodeGenerator(D=64, device=device)
        
        support_data, support_labels, query_data, query_labels = generator.generate_standard_episode(3, 2, 5)
        print(f"✓ Standard episode: Support {support_data.shape}, Query {query_data.shape}")
        
        support_data, support_labels, query_data, query_labels = generator.generate_anisotropic_episode(3, 2, 5, 2.0)
        print(f"✓ Anisotropic episode: Support {support_data.shape}, Query {query_data.shape}")
        
        print("Testing training module...")
        dict_bank = build_random_dictionary(64, device=device)
        adapter = BEMeGAAdapter(64, dict_bank, device=device)
        
        from train import compute_support_stats
        stats = compute_support_stats(support_data, support_labels)
        adapter_out = adapter(stats)
        print(f"✓ Adapter output: d={adapter_out.d}, r={adapter_out.r}, risk={adapter_out.risk_hat:.3f}")
        
        print("Testing evaluation module...")
        from evaluate import evaluate_bemega, evaluate_protonet
        from train import ProtoNetBaseline
        
        baseline = ProtoNetBaseline(64, device=device)
        test_episodes = [(support_data, support_labels, query_data, query_labels)]
        
        bemega_results = evaluate_bemega(adapter, test_episodes, device=device)
        protonet_results = evaluate_protonet(baseline, test_episodes, device=device)
        
        print(f"✓ BEMeGA accuracy: {bemega_results['mean_accuracy']:.3f}")
        print(f"✓ ProtoNet accuracy: {protonet_results['mean_accuracy']:.3f}")
        
        print("✓ All components working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main experimental pipeline"""
    print("Starting BEMeGA Experimental Pipeline...")
    start_time = time.time()
    
    set_seed(42)
    
    print_system_info()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    save_dir = os.path.join("..", ".research", "iteration1", "images")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(save_dir)}")
    
    print("\nRunning quick functionality test...")
    if not quick_test_run(device):
        print("Quick test failed! Aborting main experiments.")
        return False
    
    print("\n" + "=" * 80)
    print("STARTING MAIN EXPERIMENTS")
    print("=" * 80)
    
    all_results = {}
    
    try:
        print("\n🔬 Running K-mismatch robustness experiment...")
        k_results = run_k_mismatch_experiment(D=128, device=device)
        all_results['k_mismatch'] = k_results
        plot_results(k_results, "K-Mismatch Robustness", "Test k", save_dir)
        plot_diagnostics(k_results, "K-Mismatch Robustness", save_dir)
        print("✓ K-mismatch experiment completed")
        
        print("\n🔬 Running anisotropy robustness experiment...")
        aniso_results = run_anisotropy_experiment(D=128, device=device)
        all_results['anisotropy'] = aniso_results
        plot_results(aniso_results, "Anisotropy Robustness", "Anisotropy Factor", save_dir)
        plot_diagnostics(aniso_results, "Anisotropy Robustness", save_dir)
        print("✓ Anisotropy experiment completed")
        
        print("\n🔬 Running domain shift robustness experiment...")
        domain_results = run_domain_shift_experiment(D=128, device=device)
        all_results['domain_shift'] = domain_results
        plot_results(domain_results, "Domain Shift Robustness", "Domain Shift Factor", save_dir)
        plot_diagnostics(domain_results, "Domain Shift Robustness", save_dir)
        print("✓ Domain shift experiment completed")
        
        print("\n📊 Creating comprehensive summary plot...")
        create_summary_plot(all_results, save_dir)
        
        print_experiment_summary(all_results)
        
        results_file = os.path.join(save_dir, "bemega_results.json")
        with open(results_file, 'w') as f:
            json_results = {}
            for exp_name, exp_results in all_results.items():
                json_results[exp_name] = {}
                for condition, condition_results in exp_results.items():
                    json_results[exp_name][str(condition)] = {}
                    for method, method_results in condition_results.items():
                        json_results[exp_name][str(condition)][method] = {}
                        for key, value in method_results.items():
                            if isinstance(value, (list, np.ndarray)):
                                json_results[exp_name][str(condition)][method][key] = list(value)
                            elif isinstance(value, np.floating):
                                json_results[exp_name][str(condition)][method][key] = float(value)
                            else:
                                json_results[exp_name][str(condition)][method][key] = value
            
            json.dump(json_results, f, indent=2)
        print(f"Results saved to: {results_file}")
        
        if torch.cuda.is_available():
            print(f"\nFinal GPU memory usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ All experiments completed successfully in {elapsed_time:.1f} seconds!")
        
        print("\n📁 Generated files:")
        for file in os.listdir(save_dir):
            if file.endswith('.pdf') or file.endswith('.json'):
                print(f"  - {file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def set_status_stopped():
    """Set the status_enum to 'stopped' as required"""
    try:
        print("\n🛑 Setting status_enum to 'stopped'")
        
        status_file = os.path.join("..", ".research", "experiment_status.json")
        status_data = {
            "status_enum": "stopped",
            "completion_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": "BEMeGA",
            "success": True
        }
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print(f"Status file created: {status_file}")
        
    except Exception as e:
        print(f"Warning: Could not set status - {str(e)}")


if __name__ == "__main__":
    print("BEMeGA: Bound-conditioned Episodic Metric and Geometry Adapter")
    print("Robust Few-shot Prototypes Experiment")
    print("=" * 80)
    
    success = main()
    
    if success:
        set_status_stopped()
        print("\n🎉 BEMeGA experiment pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 BEMeGA experiment pipeline failed!")
        sys.exit(1)
