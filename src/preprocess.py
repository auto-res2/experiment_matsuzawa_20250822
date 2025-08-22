"""
CCAD-KD Preprocessing Module
Implements context discovery, ECDF storage, and data preparation utilities.
"""

import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision as tv
import torchvision.transforms.functional as TF

try:
    from tdigest import TDigest
except ImportError:
    TDigest = None

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    PCA = None
    MiniBatchKMeans = None

try:
    import timm
except ImportError:
    timm = None


def seed_all(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the best available device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def ensure_dir(path: str):
    """Ensure directory exists."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


@dataclass
class AugMeta:
    """Augmentation metadata for context discovery."""
    brightness: float
    contrast: float
    saturation: float
    crop_scale: float
    translation: float
    blur_sigma: float


class ECDFStore:
    """Per-context ECDF store using t-digest with fallback to reservoir sampling."""
    
    def __init__(self, max_contexts: int = 1024, min_count: int = 64, eps: float = 1e-4, reservoir_size: int = 512):
        self.use_tdigest = TDigest is not None
        self.max_contexts = max_contexts
        self.min_count = min_count
        self.eps = eps
        self.counts: Dict[Any, int] = {}
        self.ctx_store: Dict[Any, Any] = {}
        self.global_store: Any
        self.reservoir_size = reservoir_size
        
        if self.use_tdigest:
            self.global_store = TDigest()
        else:
            self.global_store = []

    def _evict_if_needed(self, ctx_id: Any):
        """Evict least frequent context if at capacity."""
        if ctx_id in self.ctx_store:
            return
        if len(self.ctx_store) < self.max_contexts:
            return
        drop = min(self.counts, key=self.counts.get)
        self.ctx_store.pop(drop, None)
        self.counts.pop(drop, None)

    def update_many(self, ctx_ids: List[Any], margins: torch.Tensor):
        """Update ECDFs with batch of margins."""
        mv = margins.detach().cpu().float().numpy()
        
        if self.use_tdigest:
            self.global_store.batch_update(mv.tolist())
        else:
            if len(self.global_store) >= 100000:
                self.global_store = self.global_store[len(self.global_store)//2:]
            self.global_store.extend(mv.tolist())
        
        by_ctx: Dict[Any, List[int]] = {}
        for i, c in enumerate(ctx_ids):
            by_ctx.setdefault(c, []).append(i)
            
        for c, idxs in by_ctx.items():
            self._evict_if_needed(c)
            vals = mv[idxs]
            
            if c not in self.ctx_store:
                if self.use_tdigest:
                    self.ctx_store[c] = TDigest()
                else:
                    self.ctx_store[c] = []
                self.counts[c] = 0
                
            if self.use_tdigest:
                self.ctx_store[c].batch_update(vals.tolist())
            else:
                buf = self.ctx_store[c]
                for v in vals.tolist():
                    if len(buf) < self.reservoir_size:
                        buf.append(v)
                    else:
                        j = random.randint(0, self.counts[c])
                        if j < self.reservoir_size:
                            buf[j] = v
                            
            self.counts[c] += len(vals)

    def quantiles(self, ctx_ids: List[Any], margins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get context and global quantiles for margins."""
        mv = margins.detach().cpu().float().numpy()
        
        if self.use_tdigest:
            q_ref = np.array([self.global_store.cdf(float(v)) for v in mv])
        else:
            if len(self.global_store) == 0:
                q_ref = np.full_like(mv, 0.5)
            else:
                g = np.array(self.global_store)
                q_ref = np.array([float((g <= v).mean()) for v in mv])
        
        q_t = []
        for i, c in enumerate(ctx_ids):
            if c in self.ctx_store and self.counts.get(c, 0) >= self.min_count:
                if self.use_tdigest:
                    q_t.append(self.ctx_store[c].cdf(float(mv[i])))
                else:
                    buf = np.array(self.ctx_store[c])
                    if len(buf) == 0:
                        q_t.append(q_ref[i])
                    else:
                        q_t.append(float((buf <= mv[i]).mean()))
            else:
                q_t.append(q_ref[i])
                
        q_t = np.clip(np.array(q_t), self.eps, 1 - self.eps)
        q_ref = np.clip(q_ref, self.eps, 1 - self.eps)
        
        device = margins.device
        return torch.tensor(q_t, device=device, dtype=torch.float32), torch.tensor(q_ref, device=device, dtype=torch.float32)


class DROWeights:
    """Distributionally Robust Optimization weights for worst-group control."""
    
    def __init__(self, eta: float = 0.1, clip: Tuple[float, float] = (0.25, 4.0), momentum: float = 0.9):
        self.eta = eta
        self.clip = clip
        self.momentum = momentum
        self.loss_ma: Dict[Any, float] = {}
        self.alpha_log: Dict[Any, float] = {}
        self.cached: Dict[Any, float] = {}

    def update(self, ctx_ids: List[Any], losses: torch.Tensor):
        """Update DRO weights based on per-context losses."""
        by_ctx: Dict[Any, List[float]] = {}
        for c, l in zip(ctx_ids, losses.detach().cpu().tolist()):
            by_ctx.setdefault(c, []).append(float(l))
            
        for c, ls in by_ctx.items():
            avg = float(np.mean(ls))
            if c not in self.loss_ma:
                self.loss_ma[c] = avg
                self.alpha_log[c] = 0.0
            else:
                self.loss_ma[c] = self.momentum * self.loss_ma[c] + (1 - self.momentum) * avg
        
        if len(self.loss_ma) == 0:
            return
            
        mean_loss = float(np.mean(list(self.loss_ma.values())))
        for c in self.loss_ma:
            self.alpha_log[c] += self.eta * (self.loss_ma[c] - mean_loss)
            
        keys = list(self.alpha_log.keys())
        a = np.array([self.alpha_log[k] for k in keys])
        w = np.exp(a - a.max())
        w = w / (w.sum() + 1e-12)
        w = np.clip(w, self.clip[0], self.clip[1])
        w = w / (w.sum() + 1e-12)
        
        self.cached = {k: float(w[i]) for i, k in enumerate(keys)}

    def get(self, ctx_ids: List[Any]) -> torch.Tensor:
        """Get DRO weights for given context IDs."""
        if len(self.cached) == 0:
            return torch.ones(len(ctx_ids), dtype=torch.float32)
        return torch.tensor([self.cached.get(c, 1.0) for c in ctx_ids], dtype=torch.float32)


class RandAugLogger:
    """Random augmentation with parameter logging for context discovery."""
    
    def __init__(self, img_size: int = 224, blur_prob: float = 0.5):
        self.img_size = img_size
        self.blur_prob = blur_prob
        self.gauss_blur = tv.transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))

    def __call__(self, img):
        """Apply augmentation and return image with metadata."""
        img = TF.resize(img, [self.img_size, self.img_size])
        
        brightness = random.uniform(1 - 0.4, 1 + 0.4)
        contrast = random.uniform(1 - 0.4, 1 + 0.4)
        saturation = random.uniform(1 - 0.4, 1 + 0.4)
        
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)
        img = TF.adjust_saturation(img, saturation)
        
        crop_scale = random.uniform(0.3, 1.0)
        
        translation = random.uniform(0.0, 0.2)
        
        blur_sigma = 0.0
        if random.random() < self.blur_prob:
            blur_sigma = random.uniform(0.1, 2.0)
            img = self.gauss_blur(img)
            
        img = TF.to_tensor(img)
        
        meta = AugMeta(
            brightness=brightness,
            contrast=contrast, 
            saturation=saturation,
            crop_scale=crop_scale,
            translation=translation,
            blur_sigma=blur_sigma
        )
        
        return img, meta


class SimpleCNNFeaturizer(nn.Module):
    """Lightweight CNN for style feature extraction."""
    
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, out_dim)
        
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        z = F.normalize(z, dim=-1)
        return z


class StyleClusterer:
    """Style clustering for context discovery."""
    
    def __init__(self, device: str = 'cpu', pca_dim: int = 64, n_clusters: int = 8, fit_max_samples: int = 50000):
        self.device = device
        self.pca_dim = pca_dim
        self.n_clusters = n_clusters
        self.fit_max_samples = fit_max_samples
        self.pca = None
        self.kmeans = None
        self.featurizer = SimpleCNNFeaturizer(out_dim=256).to(device).eval()

    @torch.no_grad()
    def featurize_batch(self, imgs: torch.Tensor) -> np.ndarray:
        """Extract features from image batch."""
        imgs = imgs.to(self.device)
        feats = self.featurizer(imgs)
        return feats.detach().cpu().numpy()

    def fit(self, loader: DataLoader, n_samples: int = 20000):
        """Fit PCA and K-means on sample of data."""
        if PCA is None or MiniBatchKMeans is None:
            print('[StyleClusterer] sklearn not available; using random clusters.')
            self.pca = None
            self.kmeans = None
            return
            
        X = []
        seen = 0
        
        try:
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    imgs = batch[0]
                else:
                    imgs = batch
                    
                feats = self.featurize_batch(imgs)
                X.append(feats)
                seen += len(feats)
                
                if seen >= n_samples:
                    break
        except Exception as e:
            print(f'[StyleClusterer] Error during fitting: {e}')
            print('[StyleClusterer] Using random clusters as fallback.')
            self.pca = None
            self.kmeans = None
            return
                
        if len(X) == 0:
            return
            
        X = np.vstack(X)[:n_samples]
        
        self.pca = PCA(n_components=min(self.pca_dim, X.shape[1]))
        X_pca = self.pca.fit_transform(X)
        
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, n_init=3)
        self.kmeans.fit(X_pca)
        
        print(f'[StyleClusterer] Fitted on {len(X)} samples, PCA dim={X_pca.shape[1]}, clusters={self.n_clusters}')

    @torch.no_grad()
    def predict_batch(self, imgs: torch.Tensor) -> List[int]:
        """Predict style clusters for image batch."""
        if self.pca is None or self.kmeans is None:
            return [random.randint(0, self.n_clusters-1) for _ in range(len(imgs))]
            
        feats = self.featurize_batch(imgs)
        feats_pca = self.pca.transform(feats)
        clusters = self.kmeans.predict(feats_pca)
        return clusters.tolist()


def discretize_aug_params(meta: AugMeta, n_bins: int = 4) -> str:
    """Discretize augmentation parameters into context ID."""
    def bin_val(val, min_val, max_val):
        return min(n_bins - 1, int((val - min_val) / (max_val - min_val) * n_bins))
    
    b_bin = bin_val(meta.brightness, 0.6, 1.4)
    c_bin = bin_val(meta.contrast, 0.6, 1.4)
    s_bin = bin_val(meta.saturation, 0.6, 1.4)
    scale_bin = bin_val(meta.crop_scale, 0.3, 1.0)
    trans_bin = bin_val(meta.translation, 0.0, 0.2)
    blur_bin = 1 if meta.blur_sigma > 0 else 0
    
    return f"aug_{b_bin}_{c_bin}_{s_bin}_{scale_bin}_{trans_bin}_{blur_bin}"


class ContextManager:
    """Unified context discovery and management."""
    
    def __init__(self, device: str = 'cpu', n_style_clusters: int = 8):
        self.device = device
        self.style_clusterer = StyleClusterer(device=device, n_clusters=n_style_clusters)
        self.fitted = False

    def fit_style_clusterer(self, loader: DataLoader):
        """Fit style clusterer on data."""
        self.style_clusterer.fit(loader)
        self.fitted = True

    def get_context_ids(self, imgs: torch.Tensor, aug_metas: List[AugMeta]) -> List[str]:
        """Get hybrid context IDs combining augmentation and style."""
        aug_contexts = [discretize_aug_params(meta) for meta in aug_metas]
        
        if self.fitted:
            style_clusters = self.style_clusterer.predict_batch(imgs)
        else:
            style_clusters = [0] * len(imgs)
            
        contexts = [f"{aug}__style_{style}" for aug, style in zip(aug_contexts, style_clusters)]
        return contexts


@torch.no_grad()
def teacher_margins(logits: torch.Tensor) -> torch.Tensor:
    """Compute teacher boundary-distance proxy (top-1 minus top-2 logit gap)."""
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def custom_collate_fn(batch):
    """Custom collate function to handle AugMeta objects."""
    imgs = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch])
    aug_metas = [item[2] for item in batch]  # Keep as list
    return imgs, targets, aug_metas


def get_cifar100_loaders(batch_size: int = 128, img_size: int = 224, num_workers: int = 4):
    """Get CIFAR-100 data loaders with augmentation logging."""
    aug_logger = RandAugLogger(img_size=img_size)
    
    class CIFAR100WithAug(tv.datasets.CIFAR100):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.aug_logger = aug_logger
            
        def __getitem__(self, idx):
            img, target = super().__getitem__(idx)
            if self.train:
                img, aug_meta = self.aug_logger(img)
                return img, target, aug_meta
            else:
                img = TF.to_tensor(TF.resize(img, [img_size, img_size]))
                aug_meta = AugMeta(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
                return img, target, aug_meta
    
    train_dataset = CIFAR100WithAug(root='./data', train=True, download=True)
    val_dataset = CIFAR100WithAug(root='./data', train=False, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)
    
    return train_loader, val_loader
