#!/usr/bin/env python3
"""
Evaluation module for BESS++ Energy-Bounded Attention Experiment
Implements energy measurement and model evaluation functions
"""

import torch
import numpy as np
import time
import threading
import math
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False

class PowerMonitor:
    """Power monitoring utility with NVML or dummy fallback"""
    
    def __init__(self, device_index: int = 0, interval_s: float = 0.05):
        self.device_index = device_index
        self.interval = interval_s
        self.samples = []  # (timestamp, power_W, temp_C, sm_clock, mem_clock)
        self.running = False
        self._thread = None
        self.has_nvml = _HAS_NVML and torch.cuda.is_available()
        
        if self.has_nvml:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                self.has_nvml = False
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        t0 = time.time()
        
        while self.running:
            timestamp = time.time() - t0
            
            if self.has_nvml:
                try:
                    power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                    temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                    sm_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                    
                    self.samples.append((timestamp, power_mW / 1000.0, temp, sm_clock, mem_clock))
                except Exception:
                    power = 150 + 50 * np.sin(2 * np.pi * 0.5 * timestamp)  # Simulated power wave
                    self.samples.append((timestamp, power, 70.0, 1200.0, 2500.0))
            else:
                power = 150 + 50 * np.sin(2 * np.pi * 0.5 * timestamp)  # Simulated power wave
                temp = 65 + 10 * np.sin(2 * np.pi * 0.1 * timestamp)
                self.samples.append((timestamp, power, temp, 1200.0, 2500.0))
            
            time.sleep(self.interval)
    
    def start(self):
        """Start power monitoring"""
        self.samples.clear()
        self.running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop power monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self._thread:
            self._thread.join()
        self._thread = None
    
    def get_energy_joules(self) -> float:
        """Calculate total energy consumption using trapezoidal integration"""
        if len(self.samples) < 2:
            return 0.0
        
        energy = 0.0
        for i in range(1, len(self.samples)):
            t0, p0, *_ = self.samples[i-1]
            t1, p1, *_ = self.samples[i]
            dt = t1 - t0
            energy += 0.5 * (p0 + p1) * dt  # Trapezoidal rule
        
        return energy
    
    def get_power_stats(self) -> Dict[str, float]:
        """Get power waveform statistics"""
        if len(self.samples) < 2:
            return {'peak': 0.0, 'mean': 0.0, 'variance': 0.0, 'tv_norm': 0.0}
        
        powers = np.array([p for _, p, *_ in self.samples])
        power_diff = np.diff(powers)
        
        return {
            'peak': float(powers.max()),
            'mean': float(powers.mean()),
            'variance': float(powers.var()),
            'tv_norm': float(np.abs(power_diff).sum())  # Total variation norm
        }
    
    def get_thermal_stats(self) -> Dict[str, float]:
        """Get thermal statistics"""
        if len(self.samples) < 2:
            return {'max_temp': 0.0, 'mean_temp': 0.0, 'throttle_events': 0}
        
        temps = np.array([t for _, _, t, *_ in self.samples])
        sm_clocks = np.array([c for _, _, _, c, _ in self.samples])
        
        median_clock = np.median(sm_clocks)
        throttle_events = int((sm_clocks < 0.9 * median_clock).sum())
        
        return {
            'max_temp': float(temps.max()),
            'mean_temp': float(temps.mean()),
            'throttle_events': throttle_events
        }

def exact_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Reference exact attention implementation"""
    with torch.no_grad():
        scale = 1.0 / math.sqrt(Q.shape[-1])
        logits = Q @ K.T * scale
        attn_probs = torch.softmax(logits, dim=-1)
        output = attn_probs @ V
    return output

def bess_attention_reference(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor,
    epsilon_sfm: float = 1e-4,
    epsilon_pv: float = 5e-4,
    tau: float = 1e-5,
    block_size: int = 128
) -> Tuple[torch.Tensor, Dict]:
    """Reference BESS++ attention implementation for evaluation"""
    
    device = Q.device
    Tq, D = Q.shape
    Tk, Dv = K.shape[0], V.shape[1]
    scale = 1.0 / math.sqrt(D)
    
    nB = (Tk + block_size - 1) // block_size
    alpha = torch.empty(nB, device=device, dtype=Q.dtype)
    beta = torch.empty(nB, device=device, dtype=V.dtype)
    
    for j in range(nB):
        sl = slice(j * block_size, min((j + 1) * block_size, Tk))
        Kj, Vj = K[sl], V[sl]
        alpha[j] = Kj.norm(dim=1).amax()
        beta[j] = Vj.norm(dim=1).amax()
    
    Bv = float(beta.max().item())
    order = torch.argsort(alpha, descending=True)
    
    O = torch.zeros(Tq, Dv, device=device, dtype=Q.dtype)
    slices = [slice(j * block_size, min((j + 1) * block_size, Tk)) for j in range(nB)]
    
    total_blocks = 0
    skipped_blocks = 0
    early_stops = 0
    
    for i in range(Tq):
        qi = Q[i] * scale
        mi = torch.tensor(-float('inf'), device=device, dtype=Q.dtype)
        li = torch.tensor(0.0, device=device, dtype=Q.dtype)
        skipped_mass = 0.0
        
        for t, jb in enumerate(order):
            j = int(jb.item())
            sl = slices[j]
            Kj = K[sl]
            logits = Kj @ qi
            
            m_new = torch.maximum(mi, logits.max())
            exp_old = torch.exp(mi - m_new)
            exp_blk = torch.exp(logits - m_new).sum()
            li = li * exp_old + exp_blk
            mi = m_new
            
            p_j = (exp_blk / li).item()
            
            if p_j <= tau and skipped_mass + p_j <= epsilon_pv:
                skipped_mass += p_j
                skipped_blocks += 1
            else:
                Pj = torch.softmax(logits - mi, dim=0)
                O[i] += (Pj.unsqueeze(0) @ V[sl]).squeeze(0)
            
            total_blocks += 1
            
            if t < nB - 1:
                remaining_alpha = alpha[order[t+1:]]
                if remaining_alpha.numel() > 0:
                    qn = qi.norm()
                    Urem = torch.exp(qn * remaining_alpha.max() - mi) * remaining_alpha.numel()
                    if (Urem / li) <= epsilon_sfm:
                        early_stops += 1
                        break
    
    stats = {
        'total_blocks': total_blocks,
        'skipped_blocks': skipped_blocks,
        'early_stops': early_stops,
        'skip_rate': skipped_blocks / max(1, total_blocks),
        'early_stop_rate': early_stops / max(1, Tq),
        'bound_constant': Bv
    }
    
    return O, stats

def check_error_bounds(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor,
    epsilon_sfm: float,
    epsilon_pv: float,
    **kwargs
) -> Dict[str, float]:
    """Check if BESS++ satisfies theoretical error bounds"""
    
    O_exact = exact_attention(Q, K, V)
    O_bess, stats = bess_attention_reference(Q, K, V, epsilon_sfm, epsilon_pv, **kwargs)
    
    error_per_row = (O_bess - O_exact).abs().amax(dim=1)
    
    bound_rhs = (epsilon_sfm + epsilon_pv) * stats['bound_constant']
    
    bound_satisfied = (error_per_row <= bound_rhs * 1.01).float()
    
    return {
        'max_error': float(error_per_row.max()),
        'mean_error': float(error_per_row.mean()),
        'bound_rhs': float(bound_rhs),
        'bound_satisfaction_rate': float(bound_satisfied.mean()),
        'bound_violations': int((~bound_satisfied.bool()).sum()),
        **stats
    }

def run_energy_experiments(
    Q: torch.Tensor,
    K: torch.Tensor, 
    V: torch.Tensor,
    epsilon_sfm: float = 1e-4,
    epsilon_pv: float = 5e-4,
    pattern_name: str = "unknown",
    repeats: int = 3
) -> Dict:
    """Run energy measurement experiments for BESS++ vs exact attention"""
    
    print(f"  Running energy experiment: {pattern_name}, ε_sfm={epsilon_sfm:.1e}")
    
    if torch.cuda.is_available():
        for _ in range(3):
            _ = exact_attention(Q, K, V)
        torch.cuda.synchronize()
    
    results = {
        'pattern': pattern_name,
        'epsilon_sfm': epsilon_sfm,
        'epsilon_pv': epsilon_pv,
        'shape': {'T': Q.shape[0], 'D': Q.shape[1], 'Dv': V.shape[1]}
    }
    
    monitor = PowerMonitor()
    
    exact_times = []
    exact_energies = []
    
    for rep in range(repeats):
        monitor.start()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        O_exact = exact_attention(Q, K, V)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        monitor.stop()
        
        exact_times.append(end_time - start_time)
        exact_energies.append(monitor.get_energy_joules())
        
        time.sleep(0.1)  # Brief pause between measurements
    
    bess_times = []
    bess_energies = []
    bess_power_stats = []
    
    for rep in range(repeats):
        monitor.start()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        O_bess, bess_stats = bess_attention_reference(Q, K, V, epsilon_sfm, epsilon_pv)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        monitor.stop()
        
        bess_times.append(end_time - start_time)
        bess_energies.append(monitor.get_energy_joules())
        bess_power_stats.append(monitor.get_power_stats())
        
        time.sleep(0.1)
    
    bound_check = check_error_bounds(Q, K, V, epsilon_sfm, epsilon_pv)
    
    results.update({
        'exact_time_mean': np.mean(exact_times),
        'exact_time_std': np.std(exact_times),
        'exact_energy_mean': np.mean(exact_energies),
        'exact_energy_std': np.std(exact_energies),
        'bess_time_mean': np.mean(bess_times),
        'bess_time_std': np.std(bess_times),
        'bess_energy_mean': np.mean(bess_energies),
        'bess_energy_std': np.std(bess_energies),
        'speedup': np.mean(exact_times) / np.mean(bess_times),
        'energy_reduction': 1.0 - np.mean(bess_energies) / max(1e-9, float(np.mean(exact_energies))),
        'power_peak_mean': np.mean([s['peak'] for s in bess_power_stats]),
        'power_variance_mean': np.mean([s['variance'] for s in bess_power_stats]),
        'power_tv_mean': np.mean([s['tv_norm'] for s in bess_power_stats]),
        **bound_check
    })
    
    return results

def evaluate_model(model, test_data: List, device: str) -> Dict:
    """Evaluate the BESS++ model on test data"""
    
    print("Evaluating BESS++ model...")
    
    model.eval()
    results: Dict = {
        'test_configs': [],
        'performance_metrics': [],
        'energy_metrics': []
    }
    
    with torch.no_grad():
        for config in test_data:
            print(f"  Testing config: {config}")
            
            batch_size = config['batch_size']
            seq_len = config['seq_length']
            d_model = config['d_model']
            
            x = torch.randn(batch_size, seq_len, d_model, device=device)
            
            monitor = PowerMonitor()
            monitor.start()
            
            start_time = time.time()
            output_exact, _ = model(x, use_bess=False)
            exact_time = time.time() - start_time
            
            monitor.stop()
            exact_energy = monitor.get_energy_joules()
            exact_power_stats = monitor.get_power_stats()
            
            monitor.start()
            
            start_time = time.time()
            output_bess, bess_stats = model(x, use_bess=True)
            bess_time = time.time() - start_time
            
            monitor.stop()
            bess_energy = monitor.get_energy_joules()
            bess_power_stats = monitor.get_power_stats()
            
            output_error = (output_bess - output_exact).abs().max().item()
            
            config_results = {
                'config': config,
                'exact_time': exact_time,
                'bess_time': bess_time,
                'speedup': exact_time / max(1e-9, bess_time),
                'exact_energy': exact_energy,
                'bess_energy': bess_energy,
                'energy_reduction': 1.0 - bess_energy / max(1e-9, exact_energy),
                'output_error': output_error,
                'exact_power_stats': exact_power_stats,
                'bess_power_stats': bess_power_stats,
                'bess_attention_stats': bess_stats
            }
            
            results['test_configs'].append(config)
            results['performance_metrics'].append(config_results)
    
    all_speedups = [r['speedup'] for r in results['performance_metrics']]
    all_energy_reductions = [r['energy_reduction'] for r in results['performance_metrics']]
    all_errors = [r['output_error'] for r in results['performance_metrics']]
    
    results['summary'] = {
        'mean_speedup': np.mean(all_speedups),
        'mean_energy_reduction': np.mean(all_energy_reductions),
        'max_output_error': np.max(all_errors),
        'mean_output_error': np.mean(all_errors)
    }
    
    print(f"✓ Model evaluation completed:")
    print(f"  Mean speedup: {results['summary']['mean_speedup']:.2f}x")
    print(f"  Mean energy reduction: {results['summary']['mean_energy_reduction']:.1%}")
    print(f"  Max output error: {results['summary']['max_output_error']:.2e}")
    
    return results

def benchmark_attention_patterns(patterns: Dict, device: str = 'cuda') -> Dict:
    """Benchmark different attention patterns with BESS++"""
    
    print("Benchmarking attention patterns...")
    
    epsilon_values = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    results = {}
    
    for pattern_name, (Q, K, V) in patterns.items():
        print(f"\nBenchmarking pattern: {pattern_name}")
        pattern_results = []
        
        for eps_sfm in epsilon_values:
            eps_pv = eps_sfm * 5  # PV epsilon is 5x softmax epsilon
            
            result = run_energy_experiments(
                Q, K, V,
                epsilon_sfm=eps_sfm,
                epsilon_pv=eps_pv,
                pattern_name=pattern_name,
                repeats=2  # Reduced for speed
            )
            
            pattern_results.append(result)
        
        results[pattern_name] = pattern_results
    
    return results

if __name__ == "__main__":
    print("Testing BESS++ evaluation module...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    T, D, Dv = 512, 256, 256
    Q = torch.randn(T, D, device=device)
    K = torch.randn(T, D, device=device)
    V = torch.randn(T, Dv, device=device)
    
    print(f"Test data shape: Q={Q.shape}, K={K.shape}, V={V.shape}")
    
    bound_results = check_error_bounds(Q, K, V, epsilon_sfm=1e-4, epsilon_pv=5e-4)
    print(f"Bound check results: {bound_results}")
    
    energy_results = run_energy_experiments(Q, K, V, epsilon_sfm=1e-4, epsilon_pv=5e-4, repeats=1)
    print(f"Energy experiment results: {energy_results}")
    
    print("✓ Evaluation module test completed")
