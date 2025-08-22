"""
CCAD-KD Training Module
Implements the core training loop with Context-Conditional Adaptive Distillation.
"""

import os
import time
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from .preprocess import (
    ECDFStore, DROWeights, ContextManager, teacher_margins,
    seed_all, get_device, ensure_dir
)


class QET:
    """Quantile-Equalized Teacher: per-context affine logits transform."""
    
    def __init__(self, reg: float = 1e-3, ema: float = 0.99):
        self.s: Dict[Any, float] = {}
        self.b: Dict[Any, float] = {}
        self.reg = reg
        self.ema = ema

    def transform_batch(self, ctx_ids: List[Any], logits: torch.Tensor) -> torch.Tensor:
        """Apply per-context affine transform to teacher logits."""
        device = logits.device
        s = torch.tensor([self.s.get(c, 1.0) for c in ctx_ids], device=device, dtype=logits.dtype).view(-1, 1)
        b = torch.tensor([self.b.get(c, 0.0) for c in ctx_ids], device=device, dtype=logits.dtype).view(-1, 1)
        return logits * s + b

    def update(self, ctx_ids: List[Any], margins_ctx: torch.Tensor, margins_global: torch.Tensor):
        """Update per-context transforms based on margin statistics."""
        by_ctx: Dict[Any, List[int]] = {}
        for i, c in enumerate(ctx_ids):
            by_ctx.setdefault(c, []).append(i)
            
        for c, idx in by_ctx.items():
            if len(idx) < 16:  # Need sufficient samples
                continue
                
            mc = margins_ctx[idx].detach().cpu().numpy()
            mg = margins_global[idx].detach().cpu().numpy()
            
            mc_mean, mc_std = mc.mean(), mc.std() + 1e-6
            mg_mean, mg_std = mg.mean(), mg.std() + 1e-6
            
            target_s = mg_std / mc_std
            target_b = mg_mean - target_s * mc_mean
            
            s = self.s.get(c, 1.0)
            b = self.b.get(c, 0.0)
            
            s = self.ema * s + (1 - self.ema) * target_s
            b = self.ema * b + (1 - self.ema) * target_b
            
            s = s - (1 - self.ema) * self.reg * (s - 1)
            b = b - (1 - self.ema) * self.reg * b
            
            self.s[c] = float(s)
            self.b[c] = float(b)


class KDComponents:
    """CCAD-KD core components for adaptive temperature and weighting."""
    
    def __init__(self, T_min: float = 2.0, T_max: float = 8.0, gamma: float = 1.2, 
                 beta: float = 1.5, delta: float = 1e-3, w_clip: Tuple[float, float] = (0.25, 4.0)):
        self.T_min = T_min
        self.T_max = T_max
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.w_clip = w_clip

    def adaptive_T(self, q_t: torch.Tensor) -> torch.Tensor:
        """Context-adaptive teacher temperature based on quantiles."""
        T = self.T_min + (self.T_max - self.T_min) * (1.0 - q_t).pow(self.gamma)
        return torch.clamp(T, min=self.T_min, max=self.T_max)

    def w_ctx(self, q_t: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
        """Intra-context weights from ECDF quantiles."""
        w = ((q_ref + self.delta) / (q_t + self.delta)).pow(self.beta)
        return torch.clamp(w, self.w_clip[0], self.w_clip[1])


class CCADTrainer:
    """CCAD-KD trainer with full pipeline."""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, 
                 device: str = 'cuda', warmup_epochs: int = 5, 
                 kd_alpha: float = 0.7, ce_alpha: float = 0.3):
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.kd_alpha = kd_alpha
        self.ce_alpha = ce_alpha
        
        self.ecdf_store = ECDFStore()
        self.dro_weights = DROWeights()
        self.qet = QET()
        self.kd_components = KDComponents()
        self.context_manager = ContextManager(device=device)
        
        self.epoch = 0
        self.warmup_done = False
        self.qet_enabled = False
        
        for p in self.teacher.parameters():
            p.requires_grad = False

    def _get_schedule_factors(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Get scheduling factors for CCAD components."""
        if epoch < self.warmup_epochs:
            return {'beta': 0.0, 'lambda': 0.0, 'T_range': 0.0, 'qet': 0.0}
        
        progress = (epoch - self.warmup_epochs) / max(1, total_epochs - self.warmup_epochs)
        cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
        ramp_factor = 1.0 - cosine_factor
        
        return {
            'beta': ramp_factor * 1.5,
            'lambda': ramp_factor * 0.1, 
            'T_range': ramp_factor * 6.0,  # T_max - T_min
            'qet': 1.0 if epoch > self.warmup_epochs + 2 else 0.0
        }

    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                   epoch: int, total_epochs: int) -> Dict[str, float]:
        """Train one epoch with CCAD-KD."""
        self.student.train()
        self.epoch = epoch
        
        factors = self._get_schedule_factors(epoch, total_epochs)
        self.kd_components.beta = factors['beta']
        self.dro_weights.eta = factors['lambda']
        self.kd_components.T_max = self.kd_components.T_min + factors['T_range']
        self.qet_enabled = factors['qet'] > 0.5
        
        if epoch == 0 and not self.context_manager.fitted:
            print("Fitting style clusterer...")
            self.context_manager.fit_style_clusterer(train_loader)
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')
        
        for batch_idx, (imgs, targets, aug_metas) in enumerate(pbar):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            
            ctx_ids = self.context_manager.get_context_ids(imgs, aug_metas)
            
            with torch.no_grad():
                teacher_logits = self.teacher(imgs)
                teacher_margins_batch = teacher_margins(teacher_logits)
            
            student_logits = self.student(imgs)
            
            if epoch < self.warmup_epochs:
                self.ecdf_store.update_many(ctx_ids, teacher_margins_batch)
                
            ce_loss = F.cross_entropy(student_logits, targets)
            
            if epoch >= self.warmup_epochs and len(self.ecdf_store.ctx_store) > 0:
                q_t, q_ref = self.ecdf_store.quantiles(ctx_ids, teacher_margins_batch)
                
                w_ctx = self.kd_components.w_ctx(q_t, q_ref)
                T_adaptive = self.kd_components.adaptive_T(q_t)
                
                if self.qet_enabled:
                    teacher_logits = self.qet.transform_batch(ctx_ids, teacher_logits)
                    self.qet.update(ctx_ids, teacher_margins_batch, teacher_margins_batch)
                
                kd_losses = []
                for i in range(len(imgs)):
                    T_i = T_adaptive[i].item()
                    teacher_soft = F.softmax(teacher_logits[i:i+1] / T_i, dim=1)
                    student_soft = F.log_softmax(student_logits[i:i+1] / T_i, dim=1)
                    kd_loss_i = F.kl_div(student_soft, teacher_soft, reduction='sum') * (T_i ** 2)
                    kd_losses.append(kd_loss_i * w_ctx[i])
                
                kd_loss_batch = torch.stack(kd_losses)
                
                self.dro_weights.update(ctx_ids, kd_loss_batch)
                alpha_dro = self.dro_weights.get(ctx_ids).to(self.device)
                
                kd_loss = (alpha_dro * kd_loss_batch).mean()
            else:
                T = 4.0
                teacher_soft = F.softmax(teacher_logits / T, dim=1)
                student_soft = F.log_softmax(student_logits / T, dim=1)
                kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)
            
            total_loss_batch = self.ce_alpha * ce_loss + self.kd_alpha * kd_loss
            
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            batch_size = imgs.size(0)
            total_loss += total_loss_batch.item() * batch_size
            total_ce_loss += ce_loss.item() * batch_size
            total_kd_loss += kd_loss.item() * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'KD': f'{kd_loss.item():.4f}',
                'Beta': f'{factors["beta"]:.3f}',
                'Lambda': f'{factors["lambda"]:.3f}'
            })
        
        return {
            'total_loss': total_loss / total_samples,
            'ce_loss': total_ce_loss / total_samples,
            'kd_loss': total_kd_loss / total_samples,
            'beta': factors['beta'],
            'lambda': factors['lambda'],
            'qet_enabled': self.qet_enabled
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate student model."""
        self.student.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, targets, _ in val_loader:
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.student(imgs)
            loss = F.cross_entropy(outputs, targets)
            
            total_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += imgs.size(0)
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total
        
        return {'val_loss': avg_loss, 'val_accuracy': accuracy}

    def save_checkpoint(self, filepath: str, epoch: int, optimizer: torch.optim.Optimizer, 
                       train_metrics: Dict, val_metrics: Dict):
        """Save training checkpoint."""
        ensure_dir(os.path.dirname(filepath))
        
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'ecdf_counts': dict(self.ecdf_store.counts),
            'dro_loss_ma': dict(self.dro_weights.loss_ma),
            'qet_s': dict(self.qet.s),
            'qet_b': dict(self.qet.b)
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")


def create_models(num_classes: int = 100, device: str = 'cuda'):
    """Create teacher and student models for CIFAR-100."""
    try:
        import timm
        teacher = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
        student = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
    except ImportError:
        import torchvision.models as models
        teacher = models.resnet50(pretrained=True)
        teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
        
        student = models.resnet18(pretrained=False)
        student.fc = nn.Linear(student.fc.in_features, num_classes)
    
    return teacher.to(device), student.to(device)


def train_ccad_kd(train_loader: DataLoader, val_loader: DataLoader, 
                  num_epochs: int = 50, lr: float = 0.1, device: str = 'cuda',
                  save_dir: str = './models') -> Dict[str, List[float]]:
    """Main CCAD-KD training function."""
    teacher, student = create_models(num_classes=100, device=device)
    
    trainer = CCADTrainer(teacher, student, device=device, warmup_epochs=5)
    
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {
        'train_loss': [], 'train_ce_loss': [], 'train_kd_loss': [],
        'val_loss': [], 'val_accuracy': [], 'beta': [], 'lambda': []
    }
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader, optimizer, epoch, num_epochs)
        
        val_metrics = trainer.validate(val_loader)
        
        scheduler.step()
        
        history['train_loss'].append(train_metrics['total_loss'])
        history['train_ce_loss'].append(train_metrics['ce_loss'])
        history['train_kd_loss'].append(train_metrics['kd_loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_accuracy'].append(val_metrics['val_accuracy'])
        history['beta'].append(train_metrics['beta'])
        history['lambda'].append(train_metrics['lambda'])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_metrics['total_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.2f}%")
        print(f"Beta: {train_metrics['beta']:.3f}, Lambda: {train_metrics['lambda']:.3f}, QET: {train_metrics['qet_enabled']}")
        
        if val_metrics['val_accuracy'] > best_acc:
            best_acc = val_metrics['val_accuracy']
            trainer.save_checkpoint(
                os.path.join(save_dir, 'best_model.pth'),
                epoch, optimizer, train_metrics, val_metrics
            )
        
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                epoch, optimizer, train_metrics, val_metrics
            )
    
    print(f"Training completed. Best validation accuracy: {best_acc:.2f}%")
    return history
