#!/usr/bin/env python3
"""
Training module for BESS++ Energy-Bounded Attention Experiment
Implements the BESSAttentionModel and training procedures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False

try:
    from scipy.optimize import nnls
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from sklearn.linear_model import LinearRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    _HAS_SKOPT = True
except ImportError:
    _HAS_SKOPT = False

@dataclass
class BESSConfig:
    """Configuration for BESS++ attention"""
    epsilon_sfm: float = 1e-4
    epsilon_pv: float = 5e-4
    tau: float = 1e-5
    block_size: int = 128
    projections: int = 4
    topk_rem: int = 16
    predication: bool = True
    phase_offset: float = 0.0
    occupancy: float = 1.0

class BESSAttention(nn.Module):
    """
    BESS++ Energy-Bounded Attention Module
    Implements safe early stopping and PV masking with theoretical guarantees
    """
    
    def __init__(self, d_model: int, config: Optional[BESSConfig] = None):
        super().__init__()
        self.d_model = d_model
        self.config = config or BESSConfig()
        self.scale = 1.0 / math.sqrt(d_model)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def blockwise_norms(self, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute per-block norms for BESS++ bounds"""
        T, D = K.shape
        block = self.config.block_size
        nB = (T + block - 1) // block
        
        alpha = torch.empty(nB, device=K.device, dtype=K.dtype)
        beta = torch.empty(nB, device=V.device, dtype=V.dtype)
        
        for j in range(nB):
            sl = slice(j * block, min((j + 1) * block, T))
            Kj, Vj = K[sl], V[sl]
            alpha[j] = Kj.norm(dim=1).amax()
            beta[j] = Vj.norm(dim=1).amax()
            
        return alpha, beta, nB
    
    def bess_forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """BESS++ forward pass with energy bounds"""
        device = Q.device
        Tq, D = Q.shape
        Tk, Dv = K.shape[0], V.shape[1]
        
        alpha, beta, nB = self.blockwise_norms(K, V)
        Bv = float(beta.max().item())
        
        order = torch.argsort(alpha, descending=True)
        
        O = torch.zeros(Tq, Dv, device=device, dtype=Q.dtype)
        slices = [slice(j * self.config.block_size, min((j + 1) * self.config.block_size, Tk)) for j in range(nB)]
        
        total_blocks_processed = 0
        total_skipped_blocks = 0
        total_skipped_mass = 0.0
        
        for i in range(Tq):
            qi = Q[i] * self.scale
            mi = torch.tensor(-float('inf'), device=device, dtype=Q.dtype)
            li = torch.tensor(0.0, device=device, dtype=Q.dtype)
            skipped_cum = 0.0
            processed_mask = torch.zeros(nB, device=device, dtype=torch.bool)
            
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
                
                do_pv = True
                if p_j <= self.config.tau and skipped_cum + p_j <= self.config.epsilon_pv:
                    do_pv = False
                    skipped_cum += p_j
                    total_skipped_blocks += 1
                    total_skipped_mass += p_j
                
                if do_pv:
                    Pj = torch.softmax(logits - mi, dim=0)
                    O[i] += (Pj.unsqueeze(0) @ V[sl]).squeeze(0)
                
                processed_mask[j] = True
                total_blocks_processed += 1
                
                if t < nB - 1:
                    remaining = (~processed_mask).nonzero(as_tuple=False).flatten()
                    if remaining.numel() > 0:
                        rem_sorted = remaining[torch.argsort(alpha[remaining], descending=True)]
                        rem_top = rem_sorted[:min(self.config.topk_rem, rem_sorted.numel())]
                        qn = qi.norm()
                        Urem_top = torch.exp(qn * alpha[rem_top] - mi).sum()
                        scale_factor = float(remaining.numel()) / max(1, rem_top.numel())
                        Urem = Urem_top * scale_factor
                        if (Urem / li) <= self.config.epsilon_sfm:
                            break
        
        stats = {
            'avg_blocks_processed_per_row': total_blocks_processed / max(1.0, float(Tq)),
            'blocks_total_per_row': float(nB),
            'skip_blocks_frac': total_skipped_blocks / max(1.0, float(total_blocks_processed + (Tq * nB - total_blocks_processed))),
            'skip_mass_total': total_skipped_mass,
            'bound_constant': Bv
        }
        
        return O, stats
    
    def forward(self, x: torch.Tensor, use_bess: bool = True) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with optional BESS++ optimization"""
        batch_size, seq_len, d_model = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, d_model)
        K = self.k_proj(x).view(batch_size, seq_len, d_model)
        V = self.v_proj(x).view(batch_size, seq_len, d_model)
        
        if use_bess and batch_size == 1:  # BESS++ currently supports single batch
            Q_single = Q.squeeze(0)
            K_single = K.squeeze(0)
            V_single = V.squeeze(0)
            
            O_single, stats = self.bess_forward(Q_single, K_single, V_single)
            O = O_single.unsqueeze(0)
        else:
            logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn_probs = F.softmax(logits, dim=-1)
            O = torch.matmul(attn_probs, V)
            stats = {'method': 'standard_attention'}
        
        output = self.out_proj(O)
        
        return output, stats

class BESSAttentionModel(nn.Module):
    """
    Complete BESS++ Attention Model for energy-bounded experiments
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, max_seq_len: int = 2048, device: str = 'cuda'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.attention_layers = nn.ModuleList([
            BESSAttention(d_model // n_heads) for _ in range(n_heads)
        ])
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        self.to(device)
    
    def forward(self, x: torch.Tensor, use_bess: bool = True) -> Tuple[torch.Tensor, List[Dict]]:
        """Forward pass through the model"""
        batch_size, seq_len, _ = x.shape
        
        pos_emb = self.pos_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb
        
        head_outputs = []
        head_stats = []
        
        head_dim = self.d_model // self.n_heads
        x_heads = x.view(batch_size, seq_len, self.n_heads, head_dim)
        
        for i, attn_layer in enumerate(self.attention_layers):
            head_input = x_heads[:, :, i, :].contiguous()
            head_output, stats = attn_layer(head_input, use_bess=use_bess)
            head_outputs.append(head_output)
            head_stats.append(stats)
        
        attn_output = torch.cat(head_outputs, dim=-1)
        
        x = self.layer_norm1(x + attn_output)
        
        ff_output = self.feedforward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x, head_stats

class EPRModel:
    """
    Energy Performance Regression (EPR++) Model
    Predicts energy consumption from hardware counters and configuration
    """
    
    def __init__(self):
        self.coefficients = None
        self.feature_names = [
            'wgmma_ops', 'mfu_ops', 'tma_ops', 'ldsm_ops', 'stsm_ops',
            'block_size', 'occupancy', 'phase_offset', 'temperature', 'sm_clock'
        ]
        self.is_fitted = False
    
    def extract_features(self, config: Dict, hardware_counters: Dict) -> np.ndarray:
        """Extract features for energy prediction"""
        features = []
        
        features.append(hardware_counters.get('wgmma_ops', 1000))
        features.append(hardware_counters.get('mfu_ops', 500))
        features.append(hardware_counters.get('tma_ops', 200))
        features.append(hardware_counters.get('ldsm_ops', 100))
        features.append(hardware_counters.get('stsm_ops', 50))
        
        features.append(config.get('block_size', 128))
        features.append(config.get('occupancy', 1.0))
        features.append(config.get('phase_offset', 0.0))
        features.append(config.get('temperature', 70.0))
        features.append(config.get('sm_clock', 1200.0))
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the energy model using NNLS or linear regression"""
        if _HAS_SCIPY:
            self.coefficients, _ = nnls(X, y)
        elif _HAS_SKLEARN:
            model = LinearRegression(positive=True)
            model.fit(X, y)
            self.coefficients = model.coef_
        else:
            self.coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict energy consumption"""
        if not self.is_fitted:
            return np.ones(X.shape[0]) * 1.0
        
        return X @ self.coefficients

class BayesianOptimizer:
    """
    Bayesian Optimization for BESS++ hyperparameter tuning
    """
    
    def __init__(self, search_space: Dict):
        self.search_space = search_space
        self.has_skopt = _HAS_SKOPT
        
        if self.has_skopt:
            self.dimensions = []
            self.param_names = []
            
            for name, (low, high, param_type) in search_space.items():
                self.param_names.append(name)
                if param_type == 'real':
                    self.dimensions.append(Real(low, high, name=name))
                elif param_type == 'int':
                    self.dimensions.append(Integer(low, high, name=name))
                elif param_type == 'categorical':
                    self.dimensions.append(Categorical(low, name=name))  # low is the list of categories
    
    def optimize(self, objective_func, n_calls: int = 20) -> Dict:
        """Run Bayesian optimization"""
        if not self.has_skopt:
            return self._random_search(objective_func, n_calls)
        
        try:
            result = gp_minimize(
                func=objective_func,
                dimensions=self.dimensions,
                n_calls=n_calls,
                random_state=42
            )
            
            best_params = {}
            for i, name in enumerate(self.param_names):
                best_params[name] = result.x[i]
            
            return {
                'best_params': best_params,
                'best_value': result.fun,
                'convergence': result.func_vals
            }
        except Exception as e:
            print(f"Bayesian optimization failed: {e}, falling back to random search")
            return self._random_search(objective_func, n_calls)
    
    def _random_search(self, objective_func, n_calls: int) -> Dict:
        """Fallback random search"""
        best_value = float('inf')
        best_params = {}
        convergence = []
        
        for _ in range(n_calls):
            params = {}
            for name, (low, high, param_type) in self.search_space.items():
                if param_type == 'real':
                    params[name] = np.random.uniform(low, high)
                elif param_type == 'int':
                    params[name] = np.random.randint(low, high + 1)
                elif param_type == 'categorical':
                    params[name] = np.random.choice(low)  # low is the list of categories
            
            value = objective_func([params[name] for name in self.param_names])
            convergence.append(value)
            
            if value < best_value:
                best_value = value
                best_params = params.copy()
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'convergence': convergence
        }

def train_model(model: BESSAttentionModel, train_data: List, val_data: List, device: str) -> Dict:
    """
    Train the BESS++ model with EPR++ autotuning
    """
    print("Starting BESS++ model training with EPR++ autotuning...")
    
    epr_model = EPRModel()
    
    search_space = {
        'epsilon_sfm': (1e-6, 1e-3, 'real'),
        'epsilon_pv': (1e-5, 5e-3, 'real'),
        'block_size': (64, 256, 'int'),
        'occupancy': (0.5, 1.0, 'real'),
        'phase_offset': (0.0, 0.5, 'real')
    }
    
    optimizer = BayesianOptimizer(search_space)
    
    X_epr: List[np.ndarray] = []
    y_epr: List[float] = []
    
    def objective_function(params):
        """Objective function for Bayesian optimization"""
        epsilon_sfm, epsilon_pv, block_size, occupancy, phase_offset = params
        
        config = BESSConfig(
            epsilon_sfm=epsilon_sfm,
            epsilon_pv=epsilon_pv,
            block_size=int(block_size),
            occupancy=occupancy,
            phase_offset=phase_offset
        )
        
        for attn_layer in model.attention_layers:
            attn_layer.config = config
        
        total_energy = 0.0
        total_tokens = 0
        
        for data_config in train_data[:3]:  # Use subset for efficiency
            batch_size = data_config['batch_size']
            seq_len = data_config['seq_length']
            d_model = data_config['d_model']
            
            x = torch.randn(batch_size, seq_len, d_model, device=device)
            
            start_time = time.time()
            with torch.no_grad():
                output, stats = model(x, use_bess=True)
            elapsed = time.time() - start_time
            
            simulated_power = 200 * occupancy * (1 + 0.1 * phase_offset)  # Watts
            energy = simulated_power * elapsed
            tokens = batch_size * seq_len
            
            total_energy += energy
            total_tokens += tokens
            
            hw_counters = {
                'wgmma_ops': seq_len * d_model // block_size,
                'mfu_ops': seq_len * 100,
                'tma_ops': seq_len * 50,
                'ldsm_ops': seq_len * 20,
                'stsm_ops': seq_len * 10
            }
            
            config_dict = {
                'block_size': block_size,
                'occupancy': occupancy,
                'phase_offset': phase_offset,
                'temperature': 70.0,
                'sm_clock': 1200.0
            }
            
            features = epr_model.extract_features(config_dict, hw_counters)
            X_epr.append(features)
            y_epr.append(energy / tokens)  # Energy per token
        
        energy_per_token = total_energy / max(1, total_tokens)
        
        return energy_per_token
    
    print("Running Bayesian optimization for hyperparameter tuning...")
    opt_result = optimizer.optimize(objective_function, n_calls=15)
    
    if X_epr and y_epr:
        X_epr_array = np.array(X_epr)
        y_epr_array = np.array(y_epr)
        epr_model.fit(X_epr_array, y_epr_array)
        print("✓ EPR model fitted successfully")
    
    best_params = opt_result['best_params']
    best_config = BESSConfig(
        epsilon_sfm=best_params['epsilon_sfm'],
        epsilon_pv=best_params['epsilon_pv'],
        block_size=int(best_params['block_size']),
        occupancy=best_params['occupancy'],
        phase_offset=best_params['phase_offset']
    )
    
    for attn_layer in model.attention_layers:
        attn_layer.config = best_config
    
    print(f"✓ Best configuration found:")
    print(f"  ε_sfm: {best_config.epsilon_sfm:.2e}")
    print(f"  ε_pv: {best_config.epsilon_pv:.2e}")
    print(f"  Block size: {best_config.block_size}")
    print(f"  Occupancy: {best_config.occupancy:.3f}")
    print(f"  Phase offset: {best_config.phase_offset:.3f}")
    
    return {
        'best_config': best_config,
        'optimization_result': opt_result,
        'epr_model': epr_model,
        'epr_data': (X_epr_array, y_epr_array) if X_epr else (None, None)
    }

if __name__ == "__main__":
    print("Testing BESS++ training module...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = BESSAttentionModel(d_model=256, n_heads=4, max_seq_len=512, device=device)
    
    batch_size, seq_len = 1, 256
    x = torch.randn(batch_size, seq_len, 256, device=device)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output, stats = model(x, use_bess=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention stats: {stats[0] if stats else 'None'}")
    
    print("✓ Training module test completed")
