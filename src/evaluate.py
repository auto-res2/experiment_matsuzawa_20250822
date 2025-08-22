"""
QELO Evaluation Module
Handles model evaluation including perplexity calculation and downstream task evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time


def calculate_perplexity(model: nn.Module, tokenizer, sequences: List[torch.Tensor], 
                        device: str = 'cuda', max_length: int = 2048) -> float:
    """Calculate perplexity on a list of tokenized sequences."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for seq in tqdm(sequences, desc="Computing perplexity"):
            if len(seq) < 2:
                continue
                
            if len(seq) > max_length:
                seq = seq[:max_length]
                
            input_ids = seq.unsqueeze(0).to(device)
            
            try:
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                num_tokens = input_ids.shape[1] - 1
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
            except Exception as e:
                print(f"Error processing sequence of length {len(seq)}: {e}")
                continue
                
    if total_tokens == 0:
        return float('inf')
        
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def evaluate_reconstruction_quality(original_weights: torch.Tensor,
                                  reconstructed_weights: torch.Tensor,
                                  X: torch.Tensor,
                                  Lambda_out: torch.Tensor) -> Dict[str, float]:
    """Evaluate reconstruction quality using multiple metrics."""
    with torch.no_grad():
        error = original_weights - reconstructed_weights
        frobenius_error = torch.norm(error, 'fro').item()
        relative_frobenius = frobenius_error / torch.norm(original_weights, 'fro').item()
        
        Y_orig = X @ original_weights.t()
        Y_recon = X @ reconstructed_weights.t()
        activation_error = Y_orig - Y_recon
        
        activation_mse = torch.mean(activation_error ** 2).item()
        
        Lh = torch.sqrt(Lambda_out).view(1, -1)
        weighted_activation_error = activation_error * Lh
        weighted_activation_mse = torch.mean(weighted_activation_error ** 2).item()
        
        orig_activation_var = torch.var(Y_orig).item()
        relative_activation_mse = activation_mse / orig_activation_var if orig_activation_var > 0 else float('inf')
        
        return {
            'frobenius_error': frobenius_error,
            'relative_frobenius_error': relative_frobenius,
            'activation_mse': activation_mse,
            'weighted_activation_mse': weighted_activation_mse,
            'relative_activation_mse': relative_activation_mse,
            'snr_db': -10 * math.log10(relative_activation_mse) if relative_activation_mse > 0 else float('inf')
        }


def evaluate_quantization_stability(ptq_result, luts: Optional[nn.ModuleList] = None) -> Dict[str, float]:
    """Evaluate quantization stability metrics."""
    metrics = {}
    
    M = ptq_result.M
    unique_codes, counts = torch.unique(M, return_counts=True)
    
    total_codes = ptq_result.max_code - ptq_result.min_code + 1
    used_codes = len(unique_codes)
    code_utilization = used_codes / total_codes
    
    probs = counts.float() / counts.sum()
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12)).item()
    max_entropy = math.log2(total_codes)
    normalized_entropy = entropy / max_entropy
    
    metrics.update({
        'code_utilization': code_utilization,
        'code_entropy': entropy,
        'normalized_code_entropy': normalized_entropy,
        'num_unique_codes': used_codes,
        'total_possible_codes': total_codes
    })
    
    if luts is not None:
        total_monotonicity_violations = 0
        total_lut_range = 0.0
        
        for lut in luts:
            centers = lut.centers()
            diffs = torch.diff(centers)
            violations = (diffs <= 0).sum().item()
            total_monotonicity_violations += violations
            
            lut_range = (centers.max() - centers.min()).item()
            total_lut_range += lut_range
            
        avg_lut_range = total_lut_range / len(luts)
        
        metrics.update({
            'monotonicity_violations': total_monotonicity_violations,
            'avg_lut_range': avg_lut_range
        })
    
    return metrics


def benchmark_inference_speed(model: nn.Module, input_ids: torch.Tensor, 
                            num_runs: int = 10, warmup_runs: int = 3) -> Dict[str, float]:
    """Benchmark inference speed."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_ids)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_ids)
            
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    tokens_per_second = input_ids.shape[1] / avg_time
    
    return {
        'avg_inference_time': avg_time,
        'tokens_per_second': tokens_per_second,
        'throughput_ratio': 1.0  # Baseline ratio
    }


def evaluate_memory_usage(model: nn.Module) -> Dict[str, float]:
    """Evaluate memory usage of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_memory_mb = total_params * 4 / (1024 * 1024)
    trainable_memory_mb = trainable_params * 4 / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'param_memory_mb': param_memory_mb,
        'trainable_memory_mb': trainable_memory_mb,
        'compression_ratio': 1.0  # Will be computed relative to baseline
    }


class QELOEvaluator:
    """Main evaluator for QELO experiments."""
    
    def __init__(self, model_name: str = "EleutherAI/pythia-410m"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def load_model(self, device: str = 'cuda'):
        """Load model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=device if device != 'cpu' else None
        )
        
    def evaluate_full_pipeline(self, 
                             eval_sequences: List[torch.Tensor],
                             qelo_results: Dict,
                             original_weights: Dict[str, torch.Tensor],
                             calibration_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """Evaluate the complete QELO pipeline."""
        if self.model is None:
            self.load_model()
            
        if self.model is not None:
            device = next(self.model.parameters()).device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X, Y = calibration_data
        
        results = {
            'perplexity': {},
            'reconstruction_quality': {},
            'stability_metrics': {},
            'performance_metrics': {}
        }
        
        try:
            ppl = calculate_perplexity(self.model, self.tokenizer, eval_sequences, device)
            results['perplexity']['wikitext2'] = ppl
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            results['perplexity']['wikitext2'] = float('inf')
            
        layer_metrics = {}
        for layer_name, qelo_result in qelo_results.items():
            if layer_name in original_weights:
                orig_w = original_weights[layer_name]
                
                from .train import dequantize_weights
                Q_hat = dequantize_weights(qelo_result['ptq_result'], qelo_result['luts'])
                recon_w = Q_hat + qelo_result['A'] @ qelo_result['B'].t()
                
                S_blk, Lambda_out = qelo_result.get('S_blk'), qelo_result.get('Lambda_out')
                if S_blk is None or Lambda_out is None:
                    from .train import estimate_activation_weights
                    S_blk, Lambda_out = estimate_activation_weights(X, Y)
                    
                quality_metrics = evaluate_reconstruction_quality(orig_w, recon_w, X, Lambda_out)
                stability_metrics = evaluate_quantization_stability(qelo_result['ptq_result'], qelo_result['luts'])
                
                layer_metrics[layer_name] = {
                    'quality': quality_metrics,
                    'stability': stability_metrics,
                    'training_metrics': qelo_result['metrics']
                }
                
        results['reconstruction_quality'] = layer_metrics
        
        if eval_sequences:
            sample_input = eval_sequences[0][:512].unsqueeze(0)  # Use shorter sequence for benchmarking
            speed_metrics = benchmark_inference_speed(self.model, sample_input)
            memory_metrics = evaluate_memory_usage(self.model)
            
            results['performance_metrics'] = {
                'speed': speed_metrics,
                'memory': memory_metrics
            }
            
        return results
    
    def compare_methods(self, qelo_results: Dict, baseline_results: Dict) -> Dict:
        """Compare QELO against baseline methods."""
        comparison = {
            'relative_improvements': {},
            'statistical_significance': {}
        }
        
        if 'perplexity' in qelo_results and 'perplexity' in baseline_results:
            qelo_ppl = qelo_results['perplexity'].get('wikitext2', float('inf'))
            baseline_ppl = baseline_results['perplexity'].get('wikitext2', float('inf'))
            
            if baseline_ppl != float('inf') and baseline_ppl > 0:
                ppl_improvement = (baseline_ppl - qelo_ppl) / baseline_ppl * 100
                comparison['relative_improvements']['perplexity'] = ppl_improvement
                
        qelo_quality = qelo_results.get('reconstruction_quality', {})
        baseline_quality = baseline_results.get('reconstruction_quality', {})
        
        quality_improvements = {}
        for layer_name in qelo_quality:
            if layer_name in baseline_quality:
                qelo_metrics = qelo_quality[layer_name]['quality']
                baseline_metrics = baseline_quality[layer_name]['quality']
                
                layer_improvements = {}
                for metric_name in qelo_metrics:
                    if metric_name in baseline_metrics:
                        qelo_val = qelo_metrics[metric_name]
                        baseline_val = baseline_metrics[metric_name]
                        
                        if baseline_val > 0:
                            if 'error' in metric_name or 'mse' in metric_name:
                                improvement = (baseline_val - qelo_val) / baseline_val * 100
                            else:
                                improvement = (qelo_val - baseline_val) / baseline_val * 100
                            layer_improvements[metric_name] = improvement
                            
                quality_improvements[layer_name] = layer_improvements
                
        comparison['relative_improvements']['reconstruction_quality'] = quality_improvements
        
        return comparison


def run_synthetic_evaluation():
    """Run evaluation on synthetic data for testing."""
    print("Running synthetic evaluation...")
    
    torch.manual_seed(42)
    d_in, d_out, N = 128, 64, 1000
    X = torch.randn(N, d_in)
    W = torch.randn(d_out, d_in) * 0.1
    Y = X @ W.t()
    
    from .train import QELOConfig, QELOOptimizer, estimate_activation_weights
    
    config = QELOConfig(bits=3, group_size=32, rank=8)
    optimizer = QELOOptimizer(config)
    qelo_result = optimizer.optimize_layer(W, X, Y)
    
    S_blk, Lambda_out = estimate_activation_weights(X, Y, config.group_size)
    qelo_result['S_blk'] = S_blk
    qelo_result['Lambda_out'] = Lambda_out
    
    from .train import dequantize_weights
    Q_hat = dequantize_weights(qelo_result['ptq_result'], qelo_result['luts'])
    recon_w = Q_hat + qelo_result['A'] @ qelo_result['B'].t()
    
    quality_metrics = evaluate_reconstruction_quality(W, recon_w, X, Lambda_out)
    stability_metrics = evaluate_quantization_stability(qelo_result['ptq_result'], qelo_result['luts'])
    
    print("Reconstruction Quality Metrics:")
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value:.6f}")
        
    print("\nStability Metrics:")
    for metric, value in stability_metrics.items():
        print(f"  {metric}: {value:.6f}")
        
    print("Synthetic evaluation completed successfully!")
    return quality_metrics, stability_metrics


if __name__ == "__main__":
    run_synthetic_evaluation()
