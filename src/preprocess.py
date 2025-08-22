import os
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class SyntheticSequenceDataset(Dataset):
    """Generates sequences with two patterns: 'structured' and 'noisy'."""
    def __init__(self, n_samples=2000, seq_len=32, vocab_size=256, pattern='structured', seed=0):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        rng = np.random.RandomState(seed)
        self.X = []
        self.Y = []
        if pattern == 'structured':
            blocks = 8
            block_size = vocab_size // blocks
            trans = np.zeros((vocab_size, vocab_size))
            for b in range(blocks):
                idx = slice(b * block_size, (b + 1) * block_size)
                sub = rng.dirichlet(alpha=np.ones(block_size) * 2.0)
                trans[idx, idx] = sub
            trans += 1e-2
            trans /= trans.sum(axis=1, keepdims=True)
            for _ in range(n_samples):
                seq = [int(rng.randint(0, vocab_size))]
                for _t in range(seq_len - 1):
                    p = trans[seq[-1]]
                    nxt = int(rng.choice(vocab_size, p=p))
                    seq.append(nxt)
                self.X.append(seq[:-1]); self.Y.append(seq[1:])
        else:
            for _ in range(n_samples):
                seq = rng.randint(0, vocab_size, size=seq_len)
                self.X.append(seq[:-1].tolist()); self.Y.append(seq[1:].tolist())
        self.X = torch.tensor(np.array(self.X), dtype=torch.long)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {'input_ids': self.X[idx], 'labels': self.Y[idx]}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_datasets(data_dir: str, seed: int = 0, train_n: int = 2400, val_n: int = 400):
    ensure_dir(data_dir)
    ds_train_A = SyntheticSequenceDataset(n_samples=train_n//2, pattern='structured', seed=seed)
    ds_train_B = SyntheticSequenceDataset(n_samples=train_n - train_n//2, pattern='noisy', seed=seed+1)
    ds_val_A   = SyntheticSequenceDataset(n_samples=val_n//2, pattern='structured', seed=seed+2)
    ds_val_B   = SyntheticSequenceDataset(n_samples=val_n - val_n//2, pattern='noisy', seed=seed+3)

    torch.save(ds_train_A, os.path.join(data_dir, 'train_structured.pt'))
    torch.save(ds_train_B, os.path.join(data_dir, 'train_noisy.pt'))
    torch.save(ds_val_A,   os.path.join(data_dir, 'val_structured.pt'))
    torch.save(ds_val_B,   os.path.join(data_dir, 'val_noisy.pt'))


def load_datasets(data_dir: str) -> Tuple[Dataset, Dataset]:
    ds_train_A = torch.load(os.path.join(data_dir, 'train_structured.pt'))
    ds_train_B = torch.load(os.path.join(data_dir, 'train_noisy.pt'))
    ds_val_A   = torch.load(os.path.join(data_dir, 'val_structured.pt'))
    ds_val_B   = torch.load(os.path.join(data_dir, 'val_noisy.pt'))
    train_ds = ConcatDataset([ds_train_A, ds_train_B])
    val_ds   = ConcatDataset([ds_val_A, ds_val_B])
    return train_ds, val_ds


def build_dataloaders(seed: int = 0, train_n: int = 2400, val_n: int = 400, batch_size_train: int = 32, batch_size_val: int = 64) -> Tuple[DataLoader, DataLoader]:
    # Create in-memory synthetic datasets without saving
    ds_train_A = SyntheticSequenceDataset(n_samples=train_n//2, pattern='structured', seed=seed)
    ds_train_B = SyntheticSequenceDataset(n_samples=train_n - train_n//2, pattern='noisy', seed=seed+1)
    ds_val_A   = SyntheticSequenceDataset(n_samples=val_n//2, pattern='structured', seed=seed+2)
    ds_val_B   = SyntheticSequenceDataset(n_samples=val_n - val_n//2, pattern='noisy', seed=seed+3)
    train_ds = ConcatDataset([ds_train_A, ds_train_B])
    val_ds   = ConcatDataset([ds_val_A, ds_val_B])
    train_dl = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False)
    return train_dl, val_dl
