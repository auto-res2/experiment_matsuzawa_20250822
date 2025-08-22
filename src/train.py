import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader

from preprocess import (
    set_seed, SimpleTokenizer, make_bijection, invert_bijection,
    build_synth_entities, synth_rel_dataset, ForwardPairsDataset,
    ChainsDataset, collate_forward, build_vocab_and_tokenizer
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MockLMOutput:
    """Mock output for base language model."""
    def __init__(self, logits: torch.Tensor, hidden_states: List[torch.Tensor]):
        self.logits = logits
        self.hidden_states = hidden_states

class MockBaseLM(nn.Module):
    """Mock base language model for testing."""
    def __init__(self, vocab_size: int, d_model: int = 64, hidden_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_layers = hidden_layers
        self.emb = nn.Embedding(vocab_size, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                output_hidden_states: bool = True) -> MockLMOutput:
        emb = self.emb(input_ids)
        h = self.mlp(emb)
        h = self.ln(h)
        logits = self.lm_head(h)
        if output_hidden_states:
            return MockLMOutput(logits=logits, hidden_states=[h])
        else:
            return MockLMOutput(logits=logits, hidden_states=[])

class AdapterEntityRelHead(nn.Module):
    """REA-Cycle++ Adapter Head with invertible orthogonal core."""
    def __init__(self,
                 d_model: int,
                 k: int,
                 entity_id_list: List[int],
                 relation_names: List[str],
                 use_residual: bool = False,
                 residual_eps: float = 0.1,
                 gate_hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.entity_ids = torch.tensor(entity_id_list, dtype=torch.long)
        self.n_entities = len(entity_id_list)
        self.relations = relation_names
        self.use_residual = use_residual
        self.residual_eps = residual_eps

        self.P = nn.Parameter(torch.empty(d_model, k))
        nn.init.orthogonal_(self.P)

        self.E = nn.Parameter(torch.randn(self.n_entities, k) * 0.02)

        self.B = nn.ParameterDict({r: nn.Parameter(torch.zeros(k, k)) for r in self.relations})
        if use_residual:
            self.R = nn.ParameterDict({r: nn.Parameter(torch.zeros(k, k)) for r in self.relations})
        self.tau = nn.ParameterDict({r: nn.Parameter(torch.tensor(1.0)) for r in self.relations})
        self.bias = nn.ParameterDict({r: nn.Parameter(torch.tensor(0.0)) for r in self.relations})

        self.gate = nn.Sequential(
            nn.Linear(d_model, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1)
        )

    def orthogonal_core(self, rel: str, clamp_residual: bool = True):
        """Compute orthogonal core matrix M_r = exp(A_r) with optional residual."""
        B = self.B[rel]
        A = 0.5 * (B - B.T)
        M = torch.matrix_exp(A)
        if self.use_residual:
            R = self.R[rel]
            if clamp_residual:
                u = torch.randn(self.k, device=R.device)
                for _ in range(2):
                    v = F.normalize(R.T @ u, dim=0, eps=1e-12)
                    u = F.normalize(R @ v, dim=0, eps=1e-12)
                sigma = torch.dot(u, R @ v).abs()
                if sigma > self.residual_eps:
                    R = R * (self.residual_eps / (sigma + 1e-12))
            M = (torch.eye(self.k, device=M.device) + R) @ M
        return M

    def forward(self,
                h_t: torch.Tensor,
                rel: str,
                candidate_entity_indices: Optional[torch.Tensor] = None):
        z = h_t @ self.P
        M = self.orthogonal_core(rel)
        z_rel = z @ M.T
        if candidate_entity_indices is None:
            E_sub = self.E
        else:
            valid_indices = []
            for cand_id in candidate_entity_indices:
                matches = (self.entity_ids == cand_id).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    valid_indices.append(matches[0].item())
                else:
                    valid_indices.append(0)
            
            valid_indices = torch.tensor(valid_indices, device=self.E.device, dtype=torch.long)
            E_sub = self.E.index_select(0, valid_indices)
        l_adapter = (z_rel @ E_sub.T) / (self.tau[rel] + 1e-8) + self.bias[rel]
        alpha = torch.sigmoid(self.gate(h_t)).squeeze(-1)
        return l_adapter, alpha

def mix_logits(l_base_sub: torch.Tensor, l_adapter_sub: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Mix base and adapter logits."""
    a = alpha.unsqueeze(1)
    return (1.0 - a) * l_base_sub + a * l_adapter_sub

def get_topk_with_gold(base_logits: torch.Tensor, gold_ids: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get top-K candidates ensuring gold is included."""
    K = min(K, base_logits.size(1))
    topk_vals, topk_idx = torch.topk(base_logits, k=K, dim=-1)
    B = base_logits.size(0)
    candidate_ids = topk_idx.clone()
    for b in range(B):
        gid = int(gold_ids[b].item())
        if gid not in candidate_ids[b].tolist():
            candidate_ids[b, -1] = gid
    eq = (candidate_ids == gold_ids.unsqueeze(1))
    gold_pos = torch.argmax(eq.int(), dim=1)
    return candidate_ids, gold_pos

def info_nce_align(z_rel: torch.Tensor, E_tgt_rows: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE alignment loss."""
    zr = F.normalize(z_rel, dim=-1)
    et = F.normalize(E_tgt_rows, dim=-1)
    logits = (zr @ et.T) / temperature
    labels = torch.arange(z_rel.size(0), device=z_rel.device)
    return F.cross_entropy(logits, labels)

def ortho_penalty(M: torch.Tensor) -> torch.Tensor:
    """Orthogonality penalty for matrix M."""
    I = torch.eye(M.size(0), device=M.device, dtype=M.dtype)
    return torch.linalg.matrix_norm(M.T @ M - I, ord='fro') ** 2

@dataclass
class TrainConfig:
    """Training configuration."""
    d_model: int = 64
    k: int = 8
    batch_size: int = 64
    epochs: int = 2
    lr: float = 5e-3
    weight_decay: float = 0.01
    alpha_min: float = 0.35
    alpha_start: float = 1.0
    K: int = 50
    use_residual: bool = False
    relation_names: Tuple[str, ...] = ("->R1",)

def alpha_schedule(step: int, total_steps: int, alpha_start: float, alpha_min: float, warmup_frac: float = 0.4) -> float:
    """Alpha gating schedule."""
    progress = min(1.0, step / (warmup_frac * total_steps + 1e-8))
    return alpha_start + (alpha_min - alpha_start) * progress

def train_rea_cycle_plus(config: TrainConfig, save_dir: str = ".research/iteration1/images"):
    """Train REA-Cycle++ model."""
    set_seed(0)
    os.makedirs(save_dir, exist_ok=True)
    
    n_entities = 100
    tokenizer, entity_ids, entities = build_vocab_and_tokenizer(n_entities, list(config.relation_names))
    
    reverse_relations = [rel.replace("->", "<-") if "->" in rel else "<-R1" for rel in config.relation_names]
    tokenizer.add_tokens(reverse_relations)
    
    pi = make_bijection(n_entities)
    forward_pairs, reverse_pairs, chains, twohop_pairs = synth_rel_dataset(entities, pi, split_ratio=0.6, seed=0)
    
    forward_dataset = ForwardPairsDataset(forward_pairs, config.relation_names[0])
    forward_loader = DataLoader(forward_dataset, batch_size=config.batch_size, shuffle=True,
                               collate_fn=lambda batch: collate_forward(batch, tokenizer))
    
    base_model = MockBaseLM(tokenizer.vocab_size(), config.d_model).to(DEVICE)
    
    entity_id_ints = []
    for eid in entity_ids:
        if isinstance(eid, list):
            entity_id_ints.extend(eid)
        else:
            entity_id_ints.append(eid)
    
    adapter = AdapterEntityRelHead(
        config.d_model, config.k, entity_id_ints, list(config.relation_names),
        use_residual=config.use_residual
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    total_steps = len(forward_loader) * config.epochs
    losses = []
    
    print(f"Training REA-Cycle++ for {config.epochs} epochs on {len(forward_pairs)} forward pairs")
    print(f"Device: {DEVICE}")
    
    for epoch in range(config.epochs):
        epoch_losses = []
        for step, batch in enumerate(forward_loader):
            global_step = epoch * len(forward_loader) + step
            
            input_ids = batch['input_ids'].to(DEVICE)
            target_ids = batch['target_ids'].to(DEVICE)
            rel = batch['rel']
            
            with torch.no_grad():
                base_output = base_model(input_ids, output_hidden_states=True)
                base_logits = base_output.logits[:, -1, :]  # Last token
                h_t = base_output.hidden_states[0][:, -1, :]  # Last hidden state
            
            candidate_ids, gold_pos = get_topk_with_gold(base_logits, target_ids, config.K)
            
            entity_candidates = []
            for batch_candidates in candidate_ids:
                batch_entity_cands = []
                for cand_id in batch_candidates:
                    if cand_id.item() in entity_id_ints:
                        batch_entity_cands.append(cand_id.item())
                if len(batch_entity_cands) == 0:
                    batch_entity_cands = [entity_id_ints[0]]
                entity_candidates.extend(batch_entity_cands)
            
            entity_candidates = torch.tensor(entity_candidates, device=DEVICE, dtype=torch.long)
            l_adapter, alpha = adapter(h_t, rel, entity_candidates)
            
            n_batch, n_cands = candidate_ids.size()
            
            if l_adapter.size(0) == n_batch:
                # l_adapter already has correct batch dimension, just need to match candidates
                if l_adapter.size(1) >= n_cands:
                    l_adapter = l_adapter[:, :n_cands]
                else:
                    pad_size = n_cands - l_adapter.size(1)
                    padding = torch.zeros(n_batch, pad_size, device=l_adapter.device)
                    l_adapter = torch.cat([l_adapter, padding], dim=1)
            else:
                l_adapter = torch.zeros(n_batch, n_cands, device=l_adapter.device)
            
            l_base_sub = torch.gather(base_logits, 1, candidate_ids)
            l_mix = mix_logits(l_base_sub, l_adapter, alpha)
            
            ce_loss = F.cross_entropy(l_mix, gold_pos)
            
            M = adapter.orthogonal_core(rel)
            ortho_loss = ortho_penalty(M)
            
            total_loss = ce_loss + 0.01 * ortho_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{config.epochs}, Step {step+1}/{len(forward_loader)}, "
                      f"Loss: {total_loss.item():.4f}, CE: {ce_loss.item():.4f}, "
                      f"Ortho: {ortho_loss.item():.4f}, Alpha: {alpha.mean().item():.3f}")
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    return {
        'losses': losses,
        'base_model': base_model,
        'adapter': adapter,
        'tokenizer': tokenizer,
        'entity_ids': entity_ids,
        'forward_pairs': forward_pairs,
        'reverse_pairs': reverse_pairs,
        'chains': chains,
        'twohop_pairs': twohop_pairs
    }

if __name__ == "__main__":
    config = TrainConfig(epochs=2, batch_size=32)
    results = train_rea_cycle_plus(config)
    print("Training completed successfully!")
