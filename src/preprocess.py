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

def set_seed(seed: int = 0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleTokenizer:
    """Simple tokenizer for synthetic data."""
    def __init__(self, base_tokens: Optional[List[str]] = None):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        if base_tokens is not None:
            self.add_tokens(base_tokens)

    def add_tokens(self, tokens: List[str]):
        for t in tokens:
            if t not in self.token_to_id:
                self.token_to_id[t] = len(self.id_to_token)
                self.id_to_token.append(t)

    def add_special_tokens(self, d: Dict[str, List[str]]):
        toks = d.get('additional_special_tokens', [])
        self.add_tokens(toks)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.token_to_id[t] for t in tokens]
        return self.token_to_id[tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, list):
            return [self.id_to_token[i] for i in ids]
        return self.id_to_token[ids]

    def __call__(self, texts: List[str], return_tensors='pt', padding=True):
        tokenized = [t.strip().split() for t in texts]
        max_len = max(len(t) for t in tokenized) if padding and tokenized else 0
        ids = []
        attn = []
        for toks in tokenized:
            row = [self.token_to_id.get(tok, self._add_and_get(tok)) for tok in toks]
            if padding and max_len > 0:
                pad_len = max_len - len(row)
                row = row + [0] * pad_len
                attn_row = [1] * (len(row) - pad_len) + [0] * pad_len
            else:
                attn_row = [1] * len(row)
            ids.append(row)
            attn.append(attn_row)
        out = {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn, dtype=torch.long)
        }
        if return_tensors != 'pt':
            raise ValueError('SimpleTokenizer only supports return_tensors=\'pt\'')
        return out

    def _add_and_get(self, tok: str) -> int:
        self.add_tokens([tok])
        return self.token_to_id[tok]

    def vocab_size(self):
        return len(self.id_to_token)

def make_bijection(n: int) -> Dict[int, int]:
    """Create a random bijection mapping."""
    perm = torch.randperm(n).tolist()
    return {i: perm[i] for i in range(n)}

def invert_bijection(pi: Dict[int, int]) -> Dict[int, int]:
    """Invert a bijection mapping."""
    return {v: k for k, v in pi.items()}

def build_synth_entities(n: int) -> List[str]:
    """Build synthetic entity tokens."""
    return [f"<E{str(i).zfill(4)}>" for i in range(n)]

def synth_rel_dataset(entities: List[str], pi: Dict[int, int], split_ratio: float = 0.6, seed: int = 0):
    """Generate synthetic relational dataset with forward/reverse/chain splits."""
    random.seed(seed)
    n = len(entities)
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(split_ratio * n)
    train_idx = set(idx[:n_train])

    forward_pairs = [(entities[i], entities[pi[i]]) for i in train_idx]
    inv = invert_bijection(pi)
    chains = []
    for a in train_idx:
        b = pi[a]
        c = pi[b]
        chains.append((entities[a], entities[b], entities[c]))

    heldout = [i for i in range(n) if i not in train_idx]
    reverse_pairs = [(entities[pi[i]], entities[i]) for i in heldout]
    twohop_pairs = [(entities[i], entities[pi[pi[i]]]) for i in heldout]
    return forward_pairs, reverse_pairs, chains, twohop_pairs

class ForwardPairsDataset(TorchDataset):
    """Dataset for forward relation pairs."""
    def __init__(self, pairs: List[Tuple[str, str]], relation_marker: str):
        self.rows = pairs
        self.rel = relation_marker

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        a, b = self.rows[idx]
        return {
            'prompt': f"{a} {self.rel}",
            'target': b,
            'rel': self.rel
        }

class ChainsDataset(TorchDataset):
    """Dataset for composition chains."""
    def __init__(self, chains: List[Tuple[str, str, str]], relation_marker: str):
        self.rows = chains
        self.rel = relation_marker

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        a, b, c = self.rows[idx]
        return { 'a': a, 'b': b, 'c': c, 'rel': self.rel }

def collate_forward(batch, tok: SimpleTokenizer):
    """Collate function for forward pairs."""
    texts = [x['prompt'] for x in batch]
    targets = [x['target'] for x in batch]
    enc = tok(texts, return_tensors='pt', padding=True)
    target_ids = torch.tensor(tok.convert_tokens_to_ids(targets), dtype=torch.long)
    return {**enc, 'target_ids': target_ids, 'rel': batch[0]['rel']}

def build_vocab_and_tokenizer(n_entities: int, relation_names: List[str]):
    """Build vocabulary and tokenizer for synthetic data."""
    entities = build_synth_entities(n_entities)
    base_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + entities + relation_names
    tokenizer = SimpleTokenizer(base_tokens)
    entity_ids = [tokenizer.convert_tokens_to_ids(e) for e in entities]
    return tokenizer, entity_ids, entities
