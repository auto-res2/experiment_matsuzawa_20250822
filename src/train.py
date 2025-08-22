import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Timer:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.e0 = torch.cuda.Event(enable_timing=True)
            self.e1 = torch.cuda.Event(enable_timing=True)
    
    def time_ms(self, fn):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.e0.record()
            fn()
            self.e1.record()
            self.e1.synchronize()
            return float(self.e0.elapsed_time(self.e1))
        else:
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            return (t1 - t0) * 1000.0

class NVMLPowerSampler:
    def __init__(self, device_index=0, freq_hz=200):
        self.enabled = False
        self.freq = freq_hz
        self._samples = []
        self._running = False
        self.device_index = device_index
        try:
            import pynvml
            if torch.cuda.is_available():
                pynvml.nvmlInit()
                self.nvml = pynvml
                self.dev = self.nvml.nvmlDeviceGetHandleByIndex(device_index)
                self.enabled = True
        except Exception:
            self.enabled = False

    def __enter__(self):
        if not self.enabled:
            return self
        self._samples = []
        self._running = True
        import threading
        def loop():
            while self._running:
                try:
                    p_mw = self.nvml.nvmlDeviceGetPowerUsage(self.dev)
                    t = time.time()
                    self._samples.append((t, p_mw))
                except Exception:
                    pass
                time.sleep(1.0/max(1,self.freq))
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        self._running = False
        self._thread.join(timeout=1.0)
        return False

    def energy_mJ(self, t0, t1):
        if not self.enabled:
            return None
        xs = [(ts,p) for ts,p in self._samples if t0 <= ts <= t1]
        if len(xs) < 2:
            return None
        xs.sort(key=lambda x:x[0])
        area = 0.0
        for (ta,pa),(tb,pb) in zip(xs[:-1], xs[1:]):
            area += 0.5*(pa+pb)*(tb-ta)
        return area/1000.0

class DiscreteSKUManager:
    def __init__(self, attn_windows=(64,128,256), ssm_states=(4,8,16), retr_k=(0,2)):
        self.attn_windows = list(attn_windows)
        self.ssm_states = list(ssm_states)
        self.retr_k = list(retr_k)
        self.attn_ids = list(range(len(self.attn_windows)))
        self.ssm_ids = list(range(len(self.ssm_states)))
        self.retr_ids = list(range(len(self.retr_k)))
    
    def num_skus(self):
        return {"attn": len(self.attn_ids), "ssm": len(self.ssm_ids), "retr": len(self.retr_ids)}

class CostProfiler:
    def __init__(self, sku_mgr: DiscreteSKUManager, device=None, dtype=torch.float16, repeats=5):
        self.sku = sku_mgr
        self.device = device or get_device()
        self.dtype = dtype
        self.repeats = repeats
        self.lut: Dict[Any, Dict[str, Any]] = {}
        self.timer = Timer()
        self.power_enabled = NVMLPowerSampler().enabled

    def _sdpa_local(self, q, k, v, window):
        B,L,H,D = q.shape
        q_ = q.transpose(1,2)  # [B,H,L,D]
        k_ = k.transpose(1,2)  # [B,H,L,D]
        v_ = v.transpose(1,2)  # [B,H,L,D]
        
        scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(D)  # [B,H,L,L]
        
        idx = torch.arange(L, device=q.device)
        dist = idx[None,:] - idx[:,None]
        causal_mask = torch.triu(torch.ones(L, L, device=q.device), diagonal=1).bool()
        local_mask = (dist.abs() > window)
        mask = (causal_mask | local_mask)
        
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, L)
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        y = torch.matmul(attn_weights, v_)  # [B,H,L,D]
        return y.transpose(1,2)  # [B,L,H,D]

    @torch.no_grad()
    def profile_attention(self, L,B,H,D,window):
        device = self.device
        dtype = self.dtype
        q = torch.randn(B,L,H,D, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        def fn():
            _ = self._sdpa_local(q,k,v,window)
        ms = []
        mJ = []
        for _ in range(self.repeats):
            if self.power_enabled:
                with NVMLPowerSampler() as ps:
                    t0 = time.time()
                    dt = self.timer.time_ms(fn)
                    t1 = time.time()
                    e = ps.energy_mJ(t0,t1)
                    ms.append(dt)
                    mJ.append((e if e is not None else float('nan')))
            else:
                dt = self.timer.time_ms(fn)
                ms.append(dt)
        ms_arr = np.array(ms)
        return float(np.median(ms_arr)), float(np.percentile(ms_arr,95)), (np.nanmean(mJ) if len(mJ)>0 else None)

    @torch.no_grad()
    def profile_ssm(self, L,B,D,state):
        device = self.device
        dtype = self.dtype
        ksz = int(2*state+1)
        conv = nn.Conv1d(D, D, kernel_size=ksz, padding=state, groups=D).to(device).to(dtype)
        x = torch.randn(B,D,L, device=device, dtype=dtype)
        def fn():
            _ = conv(x)
        ms = []
        mJ = []
        for _ in range(self.repeats):
            if self.power_enabled:
                with NVMLPowerSampler() as ps:
                    t0=time.time()
                    dt = self.timer.time_ms(fn)
                    t1=time.time()
                    e = ps.energy_mJ(t0,t1)
                    ms.append(dt)
                    mJ.append((e if e is not None else float('nan')))
            else:
                dt = self.timer.time_ms(fn)
                ms.append(dt)
        ms_arr = np.array(ms)
        return float(np.median(ms_arr)), float(np.percentile(ms_arr,95)), (np.nanmean(mJ) if len(mJ)>0 else None)

    def profile_all(self, shapes: List[Tuple[int,int,int,int]]):
        print("[CostProfiler] Profiling LUT for shapes:", shapes)
        for (L,B,H,Dh) in shapes:
            D = H*Dh
            for i,W in enumerate(self.sku.attn_windows):
                p50,p95,e = self.profile_attention(L,B,H,Dh,W)
                self.lut[("attn", i, (L,B,H,Dh), self.dtype)] = {"p50_ms":p50, "p95_ms":p95, "mJ": e}
            for j,S in enumerate(self.sku.ssm_states):
                p50,p95,e = self.profile_ssm(L,B,D,S)
                self.lut[("ssm", j, (L,B,H,Dh), self.dtype)] = {"p50_ms":p50, "p95_ms":p95, "mJ": e}
            for k,kd in enumerate(self.sku.retr_k):
                self.lut[("retr", k, (L,B,H,Dh), self.dtype)] = {"p50_ms": 0.2*kd, "p95_ms": 0.3*kd, "mJ": (0.01*kd if kd>0 else 0.0)}
        print("[CostProfiler] LUT entries:", len(self.lut))

    def query(self, branch, sku_id, shape, dtype=None, tail=False):
        dtype = dtype or self.dtype
        rec = self.lut.get((branch, sku_id, shape, dtype))
        if rec is None:
            return {"ms": 2.0, "mJ": 1.0}
        return {"ms": rec["p95_ms"] if tail else rec["p50_ms"], "mJ": rec.get("mJ", None)}

def st_gumbel_softmax(logits, tau=1.0, dim=-1):
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(torch.clamp(u, 1e-9, 1.0-1e-9)))
    y = F.softmax((logits+g)/tau, dim=dim)
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(dim, y.argmax(dim=dim, keepdim=True), 1.0)
    return (y_hard - y).detach() + y

class RouterR(nn.Module):
    def __init__(self, d_model, n_heads, sku_mgr: DiscreteSKUManager, taps_dim=16):
        super().__init__()
        self.d = d_model
        self.h = n_heads
        self.sku = sku_mgr
        self.taps_dim = taps_dim
        self.group_embed = nn.Embedding(max(1,n_heads//2), taps_dim)
        self.stageB = nn.Linear(d_model + taps_dim, self.h*(2 + len(self.sku.attn_ids) + len(self.sku.ssm_ids)))
        self.gate_proj = nn.Linear(d_model, self.h)
        self.temperature = 1.0

    def forward(self, x):
        B,L,D = x.shape
        H = self.h
        grp_ids = torch.arange(H, device=x.device) % max(1,H//2)
        taps = self.group_embed(grp_ids).view(1,1,H,self.taps_dim).expand(B,L,H,self.taps_dim)
        xb = x.unsqueeze(2).expand(B,L,H,D)
        logits = self.stageB(torch.cat([xb, taps], dim=-1)).view(B,L,H,-1)
        start=0
        branch_logits = logits[..., start:start+2]
        start+=2
        attn_sku_logits = logits[..., start:start+len(self.sku.attn_ids)]
        start+=len(self.sku.attn_ids)
        ssm_sku_logits  = logits[..., start:start+len(self.sku.ssm_ids)]
        branch_sel = st_gumbel_softmax(branch_logits, tau=self.temperature, dim=-1)
        attn_sku_sel = st_gumbel_softmax(attn_sku_logits, tau=self.temperature, dim=-1)
        ssm_sku_sel  = st_gumbel_softmax(ssm_sku_logits, tau=self.temperature, dim=-1)
        gates = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        branch_sel_next = torch.roll(branch_sel, shifts=-1, dims=1)
        return {
            "branch_sel": branch_sel,
            "attn_sku_sel": attn_sku_sel,
            "ssm_sku_sel": ssm_sku_sel,
            "gates": gates,
            "branch_sel_tplus1": branch_sel_next,
        }

    @staticmethod
    def fragmentation_loss(one_hot):
        if one_hot.size(1) < 2:
            return one_hot.sum()*0.0
        diff = (one_hot[:,1:,:,:] - one_hot[:,:-1,:,:]).abs().sum(dim=-1)
        return diff.mean()

class LocalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d = d_model
        self.h = n_heads
        self.dh = d_model//n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, window: int):
        B,L,D = x.shape
        H=self.h
        Dh=self.dh
        q = self.q(x).view(B,L,H,Dh).transpose(1,2)  # [B,H,L,Dh]
        k = self.k(x).view(B,L,H,Dh).transpose(1,2)  # [B,H,L,Dh]
        v = self.v(x).view(B,L,H,Dh).transpose(1,2)  # [B,H,L,Dh]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,L,L]
        
        idx = torch.arange(L, device=x.device)
        dist = idx[None,:] - idx[:,None]
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        local_mask = (dist.abs() > window)
        mask = (causal_mask | local_mask)
        
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, L)
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        y = torch.matmul(attn_weights, v)  # [B,H,L,Dh]
        y = y.transpose(1,2).reshape(B,L,D)  # [B,L,D]
        return self.o(y)

class SSMBranch(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d = d_model
        self._conv_cache: Dict[int, nn.Conv1d] = {}
    
    def forward(self, x, state_size: int):
        ksz = 2*int(state_size)+1
        conv = self._conv_cache.get(ksz)
        if conv is None:
            conv = nn.Conv1d(self.d, self.d, kernel_size=ksz, padding=ksz//2, groups=self.d)
            self._conv_cache[ksz] = conv.to(x.device, x.dtype)
        return conv(x.transpose(1,2)).transpose(1,2)

class RetrievalBranch(nn.Module):
    def __init__(self, d_model, key_dim=64):
        super().__init__()
        self.key = nn.Linear(d_model, key_dim)
        self.val = nn.Linear(d_model, d_model)
        self.mix = nn.Linear(d_model, d_model)
        self._has_faiss = False
        try:
            import faiss
            self._has_faiss = True
        except Exception:
            self._has_faiss = False
        self._mem = None
    
    def build_index(self, n=4096, d=64, seed=0):
        rng = np.random.RandomState(seed)
        self._mem = rng.randn(n, d).astype('float32')
    
    def forward(self, x, k_depth: int):
        if k_depth == 0 or self._mem is None:
            return torch.zeros_like(x)
        B,L,D = x.shape
        q = self.key(x).reshape(B*L,-1).detach().float().cpu().numpy()
        diffs = q[:,None,:] - self._mem[None,:,:]
        dists = (diffs*diffs).sum(-1)
        idx = np.argpartition(dists, kth=min(k_depth, dists.shape[1]-1), axis=1)[:,:k_depth]
        scores = 1.0 / (1e-6 + np.take_along_axis(dists, idx, axis=1).mean(axis=1))
        scores = torch.from_numpy(scores).to(x.device, x.dtype).view(B,L,1)
        return self.mix(self.val(x)) * scores

class HybridBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, sku_mgr: Optional[DiscreteSKUManager]=None):
        super().__init__()
        self.d = d_model
        self.h = n_heads
        self.sku = sku_mgr or DiscreteSKUManager()
        self.router = RouterR(d_model, n_heads, self.sku)
        self.attn = LocalAttention(d_model, n_heads)
        self.ssm = SSMBranch(d_model)
        self.retr = RetrievalBranch(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
    
    def forward(self, x, scheduler=None):
        x = self.norm1(x)
        r = self.router(x)
        if scheduler is None:
            attn_id = int(r["attn_sku_sel"].argmax(-1).mode(dim=1).values[0,0].item())
            ssm_id  = int(r["ssm_sku_sel"].argmax(-1).mode(dim=1).values[0,0].item())
            y_attn = self.attn(x, window=self.sku.attn_windows[attn_id])
            y_ssm  = self.ssm(x, state_size=self.sku.ssm_states[ssm_id])
            g = r["gates"].mean(dim=2)
            y = g*y_attn + (1.0-g)*y_ssm
            return x + self.ff(y), r
        else:
            y = scheduler.run(x, r, self.attn, self.ssm, self.retr, self.sku)
            return x + self.ff(y), r

class LAMBSModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_heads=4, n_layers=6, max_len=512, sku_mgr=None):
        super().__init__()
        self.d = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.sku = sku_mgr or DiscreteSKUManager()
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([HybridBlock(d_model, n_heads, self.sku) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        
        x = self.embed(x) + self.pos_embed(pos)
        
        router_outputs = []
        for block in self.blocks:
            x, r = block(x)
            router_outputs.append(r)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits, router_outputs

def train_model(model, train_loader, val_loader, config, cost_profiler, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    train_losses = []
    val_losses = []
    latencies = []
    energies = []
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_latency = 0.0
        epoch_energy = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            with NVMLPowerSampler() as ps:
                t0 = time.time()
                logits, router_outputs = model(x)
                t1 = time.time()
                
                task_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                
                frag_loss = 0.0
                cost_loss = 0.0
                for r in router_outputs:
                    frag_loss += RouterR.fragmentation_loss(r["branch_sel"])
                    
                    B, L, H = r["branch_sel"].shape[:3]
                    shape = (L, B, H, model.d // H)
                    
                    for h in range(H):
                        attn_probs = r["attn_sku_sel"][:, :, h, :]
                        ssm_probs = r["ssm_sku_sel"][:, :, h, :]
                        
                        for i, _ in enumerate(model.sku.attn_windows):
                            cost_info = cost_profiler.query("attn", i, shape, tail=True)
                            cost_loss += attn_probs[:, :, i].sum() * cost_info["ms"] * config['lambda_cost']
                        
                        for j, _ in enumerate(model.sku.ssm_states):
                            cost_info = cost_profiler.query("ssm", j, shape, tail=True)
                            cost_loss += ssm_probs[:, :, j].sum() * cost_info["ms"] * config['lambda_cost']
                
                total_loss = task_loss + config['lambda_frag'] * frag_loss + cost_loss
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                batch_latency = (t1 - t0) * 1000
                batch_energy = ps.energy_mJ(t0, t1) or 0.0
                
                epoch_loss += total_loss.item()
                epoch_latency += batch_latency
                epoch_energy += batch_energy
                
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'lat': f'{batch_latency:.1f}ms',
                    'energy': f'{batch_energy:.1f}mJ'
                })
        
        scheduler.step()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_latency = epoch_latency / len(train_loader)
        avg_energy = epoch_energy / len(train_loader)
        
        train_losses.append(avg_train_loss)
        latencies.append(avg_latency)
        energies.append(avg_energy)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                val_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Latency: {avg_latency:.1f}ms, Energy: {avg_energy:.1f}mJ")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'latencies': latencies,
        'energies': energies
    }
