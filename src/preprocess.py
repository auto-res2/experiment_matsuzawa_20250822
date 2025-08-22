import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import os

class SyntheticLanguageDataset(Dataset):
    def __init__(self, vocab_size=1000, seq_len=128, num_samples=10000, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.data = self._generate_data()
    
    def _generate_data(self):
        data = []
        
        for _ in range(self.num_samples):
            if np.random.random() < 0.3:
                seq = self._generate_repetitive_pattern()
            elif np.random.random() < 0.5:
                seq = self._generate_arithmetic_sequence()
            else:
                seq = self._generate_random_sequence()
            
            data.append(seq)
        
        return data
    
    def _generate_repetitive_pattern(self):
        pattern_len = np.random.randint(3, 8)
        pattern = np.random.randint(1, self.vocab_size, pattern_len)
        
        seq = []
        while len(seq) < self.seq_len:
            seq.extend(pattern)
        
        return np.array(seq[:self.seq_len])
    
    def _generate_arithmetic_sequence(self):
        start = np.random.randint(1, self.vocab_size // 2)
        step = np.random.randint(1, 5)
        
        seq = []
        current = start
        for _ in range(self.seq_len):
            seq.append(current % self.vocab_size)
            current += step
        
        return np.array(seq)
    
    def _generate_random_sequence(self):
        return np.random.randint(1, self.vocab_size, self.seq_len)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

class VariableLengthDataset(Dataset):
    def __init__(self, vocab_size=1000, min_len=32, max_len=256, num_samples=5000, seed=42):
        self.vocab_size = vocab_size
        self.min_len = min_len
        self.max_len = max_len
        self.num_samples = num_samples
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.data = self._generate_variable_length_data()
    
    def _generate_variable_length_data(self):
        data = []
        
        for _ in range(self.num_samples):
            seq_len = np.random.randint(self.min_len, self.max_len + 1)
            
            if np.random.random() < 0.4:
                seq = self._generate_long_range_dependency(seq_len)
            else:
                seq = np.random.randint(1, self.vocab_size, seq_len)
            
            data.append(seq)
        
        return data
    
    def _generate_long_range_dependency(self, seq_len):
        seq = np.random.randint(1, self.vocab_size, seq_len)
        
        marker_positions = np.random.choice(seq_len // 4, size=2, replace=False)
        marker_token = self.vocab_size - 1
        
        for pos in marker_positions:
            seq[pos] = marker_token
            if pos + seq_len // 2 < seq_len:
                seq[pos + seq_len // 2] = marker_token
        
        return seq
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    
    max_len = max(x.size(0) for x in xs)
    
    padded_xs = []
    padded_ys = []
    
    for x, y in zip(xs, ys):
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            x_padded = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y_padded = torch.cat([y, torch.zeros(pad_len, dtype=torch.long)])
        else:
            x_padded = x
            y_padded = y
        
        padded_xs.append(x_padded)
        padded_ys.append(y_padded)
    
    return torch.stack(padded_xs), torch.stack(padded_ys)

def create_dataloaders(config):
    train_dataset = SyntheticLanguageDataset(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        num_samples=config['train_samples'],
        seed=config['seed']
    )
    
    val_dataset = SyntheticLanguageDataset(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        num_samples=config['val_samples'],
        seed=config['seed'] + 1
    )
    
    test_dataset = VariableLengthDataset(
        vocab_size=config['vocab_size'],
        min_len=config['seq_len'] // 2,
        max_len=config['seq_len'] * 2,
        num_samples=config['test_samples'],
        seed=config['seed'] + 2
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'] // 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def save_dataset_info(train_loader, val_loader, test_loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    train_sample = next(iter(train_loader))
    val_sample = next(iter(val_loader))
    test_sample = next(iter(test_loader))
    
    info = {
        'train_batch_shape': train_sample[0].shape,
        'val_batch_shape': val_sample[0].shape,
        'test_batch_shape': test_sample[0].shape,
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'test_batches': len(test_loader)
    }
    
    with open(f"{save_dir}/dataset_info.txt", "w") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print("Dataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return info
