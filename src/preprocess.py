import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
import random


class SyntheticSequenceDataset(Dataset):
    def __init__(self, num_samples: int = 1000, seq_len: int = 128, vocab_size: int = 50, 
                 num_classes: int = 2, pattern_complexity: str = 'medium', seed: int = 42):
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.pattern_complexity = pattern_complexity
        
        self.sequences, self.labels = self._generate_data()
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        labels = []
        
        for _ in range(self.num_samples):
            if self.pattern_complexity == 'simple':
                seq, label = self._generate_simple_pattern()
            elif self.pattern_complexity == 'medium':
                seq, label = self._generate_medium_pattern()
            else:
                seq, label = self._generate_complex_pattern()
            
            sequences.append(seq)
            labels.append(label)
        
        return torch.stack(sequences), torch.tensor(labels)
    
    def _generate_simple_pattern(self) -> Tuple[torch.Tensor, int]:
        seq = torch.randint(0, self.vocab_size, (self.seq_len,))
        
        special_token = self.vocab_size - 1
        num_special = (seq == special_token).sum().item()
        
        label = 1 if num_special > self.seq_len // 4 else 0
        return seq, label
    
    def _generate_medium_pattern(self) -> Tuple[torch.Tensor, int]:
        seq = torch.randint(0, self.vocab_size - 3, (self.seq_len,))
        
        pattern_tokens = [self.vocab_size - 3, self.vocab_size - 2, self.vocab_size - 1]
        
        if random.random() < 0.5:
            pattern_start = random.randint(0, max(0, self.seq_len - 10))
            for i, token in enumerate(pattern_tokens):
                if pattern_start + i < self.seq_len:
                    seq[pattern_start + i] = token
            label = 1
        else:
            label = 0
        
        return seq, label
    
    def _generate_complex_pattern(self) -> Tuple[torch.Tensor, int]:
        seq = torch.randint(0, self.vocab_size - 5, (self.seq_len,))
        
        pattern_a = [self.vocab_size - 5, self.vocab_size - 4]
        pattern_b = [self.vocab_size - 3, self.vocab_size - 2, self.vocab_size - 1]
        
        has_pattern_a = False
        has_pattern_b = False
        
        if random.random() < 0.4:
            pos = random.randint(0, self.seq_len - len(pattern_a))
            for i, token in enumerate(pattern_a):
                seq[pos + i] = token
            has_pattern_a = True
        
        if random.random() < 0.4:
            pos = random.randint(0, self.seq_len - len(pattern_b))
            for i, token in enumerate(pattern_b):
                seq[pos + i] = token
            has_pattern_b = True
        
        label = 1 if (has_pattern_a and has_pattern_b) else 0
        return seq, label
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'tokens': self.sequences[idx],
            'labels': self.labels[idx]
        }


class ContinuousSignalDataset(Dataset):
    def __init__(self, num_samples: int = 1000, seq_len: int = 256, num_classes: int = 2, 
                 signal_type: str = 'mixed', seed: int = 42):
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.signal_type = signal_type
        
        self.signals, self.labels = self._generate_signals()
    
    def _generate_signals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        signals = []
        labels = []
        
        for _ in range(self.num_samples):
            if self.signal_type == 'sine':
                signal, label = self._generate_sine_signal()
            elif self.signal_type == 'ar':
                signal, label = self._generate_ar_signal()
            else:
                signal, label = self._generate_mixed_signal()
            
            signals.append(signal)
            labels.append(label)
        
        return torch.stack(signals), torch.tensor(labels)
    
    def _generate_sine_signal(self) -> Tuple[torch.Tensor, int]:
        t = torch.linspace(0, 4*np.pi, self.seq_len)
        
        if random.random() < 0.5:
            freq = random.uniform(0.5, 2.0)
            phase = random.uniform(0, 2*np.pi)
            signal = torch.sin(freq * t + phase)
            label = 0
        else:
            freq1 = random.uniform(0.5, 1.5)
            freq2 = random.uniform(2.0, 4.0)
            phase1 = random.uniform(0, 2*np.pi)
            phase2 = random.uniform(0, 2*np.pi)
            signal = torch.sin(freq1 * t + phase1) + 0.5 * torch.sin(freq2 * t + phase2)
            label = 1
        
        noise = torch.randn_like(signal) * 0.1
        return signal + noise, label
    
    def _generate_ar_signal(self) -> Tuple[torch.Tensor, int]:
        if random.random() < 0.5:
            a1, a2 = 0.7, -0.2
            label = 0
        else:
            a1, a2 = -0.5, 0.3
            label = 1
        
        signal = torch.zeros(self.seq_len)
        noise = torch.randn(self.seq_len) * 0.3
        
        for t in range(2, self.seq_len):
            signal[t] = a1 * signal[t-1] + a2 * signal[t-2] + noise[t]
        
        return signal, label
    
    def _generate_mixed_signal(self) -> Tuple[torch.Tensor, int]:
        if random.random() < 0.5:
            signal, label = self._generate_sine_signal()
        else:
            signal, label = self._generate_ar_signal()
        
        return signal, label
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'signals': self.signals[idx].unsqueeze(0),
            'labels': self.labels[idx]
        }


def create_tiny_data_splits(dataset: Dataset, tiny_fraction: float = 0.05, 
                          val_fraction: float = 0.2, test_fraction: float = 0.2) -> Tuple[Dataset, Dataset, Dataset]:
    
    total_size = len(dataset)
    tiny_size = max(1, int(total_size * tiny_fraction))
    
    remaining_size = total_size - tiny_size
    val_size = int(remaining_size * val_fraction)
    test_size = int(remaining_size * test_fraction)
    train_size = tiny_size
    
    indices = torch.randperm(total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset,
                       batch_size: int = 32, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def prepare_discrete_experiment(num_samples: int = 2000, seq_len: int = 128, vocab_size: int = 50,
                              pattern_complexity: str = 'medium', tiny_fraction: float = 0.05,
                              batch_size: int = 32, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    
    print(f"Creating discrete sequence dataset with {num_samples} samples...")
    dataset = SyntheticSequenceDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        pattern_complexity=pattern_complexity,
        seed=seed
    )
    
    print(f"Creating tiny data splits (train: {tiny_fraction*100:.1f}%)...")
    train_dataset, val_dataset, test_dataset = create_tiny_data_splits(
        dataset, tiny_fraction=tiny_fraction
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )
    
    return train_loader, val_loader, test_loader, vocab_size


def prepare_continuous_experiment(num_samples: int = 2000, seq_len: int = 256, 
                                signal_type: str = 'mixed', tiny_fraction: float = 0.05,
                                batch_size: int = 32, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    print(f"Creating continuous signal dataset with {num_samples} samples...")
    dataset = ContinuousSignalDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        signal_type=signal_type,
        seed=seed
    )
    
    print(f"Creating tiny data splits (train: {tiny_fraction*100:.1f}%)...")
    train_dataset, val_dataset, test_dataset = create_tiny_data_splits(
        dataset, tiny_fraction=tiny_fraction
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )
    
    return train_loader, val_loader, test_loader


def get_data_statistics(loader: DataLoader, data_type: str = 'discrete') -> Dict:
    total_samples = 0
    label_counts = {}
    
    if data_type == 'discrete':
        vocab_counts = {}
        seq_lengths = []
        
        for batch in loader:
            tokens = batch['tokens']
            labels = batch['labels']
            
            total_samples += tokens.size(0)
            seq_lengths.extend([tokens.size(1)] * tokens.size(0))
            
            for label in labels:
                label_counts[label.item()] = label_counts.get(label.item(), 0) + 1
            
            for seq in tokens:
                for token in seq:
                    vocab_counts[token.item()] = vocab_counts.get(token.item(), 0) + 1
        
        return {
            'total_samples': total_samples,
            'label_distribution': label_counts,
            'vocab_distribution': vocab_counts,
            'avg_seq_length': np.mean(seq_lengths),
            'vocab_size': len(vocab_counts)
        }
    
    else:
        signal_stats = []
        
        for batch in loader:
            signals = batch['signals']
            labels = batch['labels']
            
            total_samples += signals.size(0)
            
            for label in labels:
                label_counts[label.item()] = label_counts.get(label.item(), 0) + 1
            
            signal_stats.extend([
                {
                    'mean': sig.mean().item(),
                    'std': sig.std().item(),
                    'min': sig.min().item(),
                    'max': sig.max().item()
                }
                for sig in signals
            ])
        
        return {
            'total_samples': total_samples,
            'label_distribution': label_counts,
            'signal_mean': np.mean([s['mean'] for s in signal_stats]),
            'signal_std': np.mean([s['std'] for s in signal_stats]),
            'signal_range': (
                np.mean([s['min'] for s in signal_stats]),
                np.mean([s['max'] for s in signal_stats])
            )
        }
