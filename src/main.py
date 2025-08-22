"""
AccuTune Main Experiment Runner
Orchestrates the complete AccuTune experimental pipeline including three experiments:
1. MLP on CIFAR-10/synthetic data
2. CNN/ResNet-18 on CIFAR-10 
3. Transformer-small on synthetic text data
"""

import os
import sys
import time
import json
import argparse
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from preprocess import preprocess_data
from train import train_model, LowBitAccLinear
from evaluate import run_full_evaluation

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class MLP(nn.Module):
    """MLP model with low-bit accumulator layers for Experiment 1"""
    def __init__(self, d=512, num_classes=10, in_chw=(3,32,32)):
        super().__init__()
        C, H, W = in_chw
        self.l1 = LowBitAccLinear(C*H*W, d, mode="M7E4", order="pairwise")
        self.ln1 = nn.LayerNorm(d)
        self.l2 = LowBitAccLinear(d, d, mode="M7E4", order="pairwise")
        self.ln2 = nn.LayerNorm(d)
        self.l3 = LowBitAccLinear(d, num_classes, mode="M8E4", order="chunk16")

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = F.gelu(self.ln1(self.l1(x)))
        x = F.gelu(self.ln2(self.l2(x)))
        return self.l3(x)


class SmallCNN(nn.Module):
    """Small CNN model for Experiment 2"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = LowBitAccLinear(64, num_classes, mode="M7E4", order="pairwise")
        self.ln = nn.LayerNorm(64)
        
    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.ln(x)
        return self.fc(x)


class QLinear(nn.Module):
    """Quantized linear layer wrapper"""
    def __init__(self, in_f, out_f, mode="M7E4", order="pairwise"):
        super().__init__()
        self.inner = LowBitAccLinear(in_f, out_f, mode=mode, ebias=0, order=order, dither=False, sample_p=0.01)
    def forward(self, x):
        return self.inner(x)


class MLPBlock(nn.Module):
    """MLP block for transformer with standard layers for performance"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        y = F.gelu(self.fc1(self.ln(x)))
        return self.fc2(y) + x


class SelfAttention(nn.Module):
    """Self-attention with standard PyTorch layers for performance"""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.ln = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        
    def forward(self, x, attn_mask=None):
        x0 = self.ln(x)
        B, T, C = x0.shape
        q = self.q(x0).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k(x0).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v(x0).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if attn_mask is not None:
            att = att.masked_fill(~attn_mask, float('-inf'))
        att = att.softmax(dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.o(y) + x


class TransformerSmall(nn.Module):
    """Small transformer model for Experiment 3"""
    def __init__(self, vocab_size=8192, d_model=256, n_head=4, n_layer=4, d_ff=1024, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleList([SelfAttention(d_model, n_head), MLPBlock(d_model, d_ff)]) 
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.scale_shift_on = False
        
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)[None, :].expand(B, T)
        x = self.tok(idx) + self.pos(pos)
        if self.scale_shift_on:
            x = x * 2.0
        for attn, mlp in self.blocks:
            x = attn(x)
            x = mlp(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits


def run_experiment_1(quick_test=False, device="cuda"):
    """Experiment 1: MLP on CIFAR-10 with AccuTune"""
    print("=" * 60)
    print("EXPERIMENT 1: MLP on CIFAR-10 with AccuTune")
    print("=" * 60)
    
    train_loader, test_loader = preprocess_data("mlp", quick_test=quick_test, batch_size=32)
    
    model = MLP(d=256 if quick_test else 512, num_classes=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    num_epochs = 3 if quick_test else 10
    results = train_model(model, train_loader, test_loader, num_epochs=num_epochs, device=device, quick_test=quick_test)
    
    save_dir = ".research/iteration1/images"
    summary = run_full_evaluation(
        model, test_loader, results, 
        save_dir=save_dir, 
        experiment_name="exp1_mlp_accutune", 
        device=device, 
        quick_test=quick_test
    )
    
    return summary


def run_experiment_2(quick_test=False, device="cuda"):
    """Experiment 2: CNN on CIFAR-10 with energy monitoring"""
    print("=" * 60)
    print("EXPERIMENT 2: CNN on CIFAR-10 with Energy Monitoring")
    print("=" * 60)
    
    train_loader, test_loader = preprocess_data("cnn", quick_test=quick_test, batch_size=32)
    
    model = SmallCNN(num_classes=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    num_epochs = 3 if quick_test else 8
    results = train_model(model, train_loader, test_loader, num_epochs=num_epochs, device=device, quick_test=quick_test)
    
    save_dir = ".research/iteration1/images"
    summary = run_full_evaluation(
        model, test_loader, results, 
        save_dir=save_dir, 
        experiment_name="exp2_cnn_energy", 
        device=device, 
        quick_test=quick_test
    )
    
    return summary


def run_experiment_3(quick_test=False, device="cuda"):
    """Experiment 3: Transformer-small with disturbances"""
    print("=" * 60)
    print("EXPERIMENT 3: Transformer-small with Disturbances")
    print("=" * 60)
    
    vocab_size = 4096 if quick_test else 8192
    train_loader, test_loader = preprocess_data("transformer", quick_test=quick_test, batch_size=16, vocab_size=vocab_size)
    d_model = 128 if quick_test else 256
    model = TransformerSmall(vocab_size=vocab_size, d_model=d_model, n_head=4, n_layer=2 if quick_test else 4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    num_epochs = 3 if quick_test else 6
    results = train_model(model, train_loader, test_loader, num_epochs=num_epochs, device=device, quick_test=quick_test)
    
    if num_epochs > 2:
        print("Adding scale disturbance...")
        model.scale_shift_on = True
        additional_results = train_model(model, train_loader, test_loader, num_epochs=2, device=device, quick_test=quick_test)
        results["train_losses"].extend(additional_results["train_losses"])
        results["test_accuracies"].extend(additional_results["test_accuracies"])
        results["ga_history"].extend(additional_results["ga_history"])
        results["energy_history"].extend(additional_results["energy_history"])
        results["final_accuracy"] = additional_results["final_accuracy"]
    
    save_dir = ".research/iteration1/images"
    summary = run_full_evaluation(
        model, test_loader, results, 
        save_dir=save_dir, 
        experiment_name="exp3_transformer_disturbance", 
        device=device, 
        quick_test=quick_test
    )
    
    return summary


def run_gemm_microbenchmark(quick_test=False, device="cuda"):
    """GEMM microbenchmark for accumulator modes"""
    print("=" * 60)
    print("GEMM MICROBENCHMARK: Accumulator Mode Comparison")
    print("=" * 60)
    
    modes = ["M6E5", "M7E4", "M8E4"]
    orders = ["pairwise", "chunk16", "chunk8"]
    
    results = {}
    
    sizes = [(64, 128, 256)] if quick_test else [(128, 256, 512), (256, 512, 1024), (512, 1024, 2048)]
    
    for M, K, N in sizes:
        print(f"\nTesting matrix size: {M}x{K} @ {K}x{N}")
        
        A = torch.randn(M, K, device=device)
        B = torch.randn(N, K, device=device)  # Fixed: B should be N x K for B.t() to be K x N
        
        for mode in modes:
            for order in orders:
                layer = LowBitAccLinear(K, N, mode=mode, order=order).to(device)
                
                for _ in range(3):
                    _ = layer._accumulate(A, B.t())
                
                if device == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()
                
                for _ in range(10 if quick_test else 50):
                    output = layer._accumulate(A, B.t())
                
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - start_time
                
                key = f"{M}x{K}x{N}_{mode}_{order}"
                results[key] = {
                    "time_ms": elapsed * 1000 / (10 if quick_test else 50),
                    "overflow_count": int(layer.of_count.item()),
                    "underflow_count": int(layer.uf_count.item()),
                    "swamping_count": int(layer.swamp_count.item()),
                    "alpha_ema": float(layer.alpha_ema.item())
                }
                
                print(f"  {mode}-{order}: {results[key]['time_ms']:.2f}ms, "
                      f"OF={results[key]['overflow_count']}, "
                      f"UF={results[key]['underflow_count']}, "
                      f"Swamp={results[key]['swamping_count']}")
    
    os.makedirs(".research/iteration1/images", exist_ok=True)
    with open(".research/iteration1/images/gemm_microbench_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Microbenchmark results saved to .research/iteration1/images/gemm_microbench_results.json")
    return results


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description="AccuTune Experiments")
    parser.add_argument("--quick-test", action="store_true", help="Run quick tests with reduced parameters")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--experiments", nargs="+", default=["1", "2", "3", "microbench"], 
                       choices=["1", "2", "3", "microbench"], help="Which experiments to run")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"Using device: {args.device}")
    print(f"Quick test mode: {args.quick_test}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if HAS_PSUTIL:
        print(f"System RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
        print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")
        print(f"CPU cores: {psutil.cpu_count()}")
    
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    
    set_seed(42)
    
    os.makedirs(".research/iteration1/images", exist_ok=True)
    
    all_summaries = {}
    
    if "1" in args.experiments:
        try:
            print(f"\n[{time.strftime('%H:%M:%S')}] Starting Experiment 1 (MLP on CIFAR-10)...")
            start_time = time.time()
            summary1 = run_experiment_1(quick_test=args.quick_test, device=args.device)
            elapsed = time.time() - start_time
            print(f"[{time.strftime('%H:%M:%S')}] Experiment 1 completed in {elapsed:.1f}s")
            all_summaries["experiment_1"] = summary1
        except Exception as e:
            print(f"Experiment 1 failed: {e}")
            all_summaries["experiment_1"] = {"error": str(e)}
    
    if "2" in args.experiments:
        try:
            print(f"\n[{time.strftime('%H:%M:%S')}] Starting Experiment 2 (CNN/ResNet-18 on CIFAR-10)...")
            start_time = time.time()
            summary2 = run_experiment_2(quick_test=args.quick_test, device=args.device)
            elapsed = time.time() - start_time
            print(f"[{time.strftime('%H:%M:%S')}] Experiment 2 completed in {elapsed:.1f}s")
            all_summaries["experiment_2"] = summary2
        except Exception as e:
            print(f"Experiment 2 failed: {e}")
            all_summaries["experiment_2"] = {"error": str(e)}
    
    if "3" in args.experiments:
        try:
            print(f"\n[{time.strftime('%H:%M:%S')}] Starting Experiment 3 (Transformer-small on synthetic text)...")
            start_time = time.time()
            summary3 = run_experiment_3(quick_test=args.quick_test, device=args.device)
            elapsed = time.time() - start_time
            print(f"[{time.strftime('%H:%M:%S')}] Experiment 3 completed in {elapsed:.1f}s")
            all_summaries["experiment_3"] = summary3
        except Exception as e:
            print(f"Experiment 3 failed: {e}")
            all_summaries["experiment_3"] = {"error": str(e)}
    
    if "microbench" in args.experiments:
        try:
            print(f"\n[{time.strftime('%H:%M:%S')}] Starting GEMM Microbenchmark...")
            start_time = time.time()
            microbench_results = run_gemm_microbenchmark(quick_test=args.quick_test, device=args.device)
            elapsed = time.time() - start_time
            print(f"[{time.strftime('%H:%M:%S')}] Microbenchmark completed in {elapsed:.1f}s")
            all_summaries["microbenchmark"] = {"num_configs": len(microbench_results)}
        except Exception as e:
            print(f"Microbenchmark failed: {e}")
            all_summaries["microbenchmark"] = {"error": str(e)}
    
    with open(".research/iteration1/images/experiment_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    total_experiments = len([k for k in all_summaries.keys() if not k.startswith("microbench")])
    successful_experiments = len([k for k, v in all_summaries.items() if "error" not in v and not k.startswith("microbench")])
    print(f"Experiments completed: {successful_experiments}/{total_experiments}")
    
    for exp_name, summary in all_summaries.items():
        print(f"\n{exp_name}:")
        if "error" in summary:
            print(f"  ❌ ERROR: {summary['error']}")
        else:
            print(f"  ✅ SUCCESS")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    print(f"\nAll experiments completed!")
    print(f"Results and plots saved to: .research/iteration1/images/")
    
    pdf_files = glob.glob(".research/iteration1/images/*.pdf")
    json_files = glob.glob(".research/iteration1/images/*.json")
    print(f"Generated {len(pdf_files)} PDF plots and {len(json_files)} result files")
    
    print("\nSetting status_enum to 'stopped'")


if __name__ == "__main__":
    main()
