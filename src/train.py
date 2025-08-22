#!/usr/bin/env python

"""
Training module for CAMoE-Diff experiment.
Implements training loop with composite loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
from tqdm import tqdm


class DiffusionTrainer:
    """Trainer for diffusion models with MoE support."""
    
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.device = config['device']
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-6)
        )
        
        self.cost_reg_lambda = config.get('cost_reg_lambda', 0.01)
        self.balance_loss_lambda = config.get('balance_loss_lambda', 0.01)
        
        self.train_losses = []
        self.val_losses = []
        self.cost_losses = []
        self.balance_losses = []
        
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adds noise to images according to diffusion schedule."""
        beta_start = 0.0001
        beta_end = 0.02
        
        betas = torch.linspace(beta_start, beta_end, self.config['timesteps'], device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t.long()])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[t.long()])
        
        noise = torch.randn_like(x)
        
        noisy_x = (
            sqrt_alphas_cumprod.view(-1, 1, 1, 1) * x +
            sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1) * noise
        )
        
        return noisy_x, noise
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Performs a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = batch.shape[0]
        
        t = torch.randint(0, self.config['timesteps'], (batch_size,), device=self.device).float()
        
        noisy_batch, target_noise = self.add_noise(batch, t)
        
        predicted_noise, aux_losses, routing_decisions = self.model(noisy_batch, t)
        
        denoising_loss = F.mse_loss(predicted_noise, target_noise)
        cost_loss = self.cost_reg_lambda * aux_losses.get('cost', 0.0)
        balance_loss = self.balance_loss_lambda * aux_losses.get('balance', 0.0)
        
        total_loss = denoising_loss + cost_loss + balance_loss
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'denoising_loss': denoising_loss.item(),
            'cost_loss': cost_loss.item() if isinstance(cost_loss, torch.Tensor) else cost_loss,
            'balance_loss': balance_loss.item() if isinstance(balance_loss, torch.Tensor) else balance_loss
        }
    
    def validate(self, val_data: torch.Tensor) -> Dict[str, float]:
        """Performs validation."""
        self.model.eval()
        total_losses = []
        
        with torch.no_grad():
            for i in range(0, len(val_data), self.config['batch_size']):
                batch = val_data[i:i+self.config['batch_size']]
                batch_size = batch.shape[0]
                
                t = torch.randint(0, self.config['timesteps'], (batch_size,), device=self.device).float()
                
                noisy_batch, target_noise = self.add_noise(batch, t)
                
                predicted_noise, aux_losses, _ = self.model(noisy_batch, t)
                
                denoising_loss = F.mse_loss(predicted_noise, target_noise)
                total_losses.append(denoising_loss.item())
        
        return {'val_loss': float(np.mean(total_losses))}
    
    def train(self, train_data: torch.Tensor, val_data: torch.Tensor, epochs: int) -> Dict[str, List[float]]:
        """Main training loop."""
        print(f"Starting training for {epochs} epochs...")
        print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_cost_losses = []
            epoch_balance_losses = []
            
            pbar = tqdm(range(0, len(train_data), self.config['batch_size']), 
                       desc=f"Epoch {epoch+1}/{epochs}")
            
            for i in pbar:
                batch = train_data[i:i+self.config['batch_size']]
                losses = self.train_step(batch)
                
                epoch_losses.append(losses['denoising_loss'])
                epoch_cost_losses.append(losses['cost_loss'])
                epoch_balance_losses.append(losses['balance_loss'])
                
                pbar.set_postfix({
                    'Loss': f"{losses['denoising_loss']:.4f}",
                    'Cost': f"{losses['cost_loss']:.4f}",
                    'Balance': f"{losses['balance_loss']:.4f}"
                })
            
            val_metrics = self.validate(val_data)
            
            self.train_losses.append(np.mean(epoch_losses))
            self.val_losses.append(val_metrics['val_loss'])
            self.cost_losses.append(np.mean(epoch_cost_losses))
            self.balance_losses.append(np.mean(epoch_balance_losses))
            
            print(f"Epoch {epoch+1}: Train Loss: {self.train_losses[-1]:.4f}, "
                  f"Val Loss: {self.val_losses[-1]:.4f}, "
                  f"Cost Loss: {self.cost_losses[-1]:.4f}, "
                  f"Balance Loss: {self.balance_losses[-1]:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'cost_losses': self.cost_losses,
            'balance_losses': self.balance_losses
        }
    
    def plot_training_curves(self, save_dir: str):
        """Plots and saves training curves."""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].plot(self.train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Denoising Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.cost_losses, label='Cost Regularization', color='green')
        axes[0, 1].set_title('Cost Regularization Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Cost Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(self.balance_losses, label='Load Balance', color='orange')
        axes[1, 0].set_title('Load Balancing Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Balance Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.train_losses, label='Denoising', alpha=0.7)
        axes[1, 1].plot(np.array(self.cost_losses) * 10, label='Cost (×10)', alpha=0.7)
        axes[1, 1].plot(np.array(self.balance_losses) * 10, label='Balance (×10)', alpha=0.7)
        axes[1, 1].set_title('All Losses (Scaled)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_dir}/training_curves.pdf")
    
    def save_model(self, save_path: str):
        """Saves the trained model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'cost_losses': self.cost_losses,
            'balance_losses': self.balance_losses
        }, save_path)
        print(f"Model saved to {save_path}")


def train_model(model, train_data: torch.Tensor, val_data: torch.Tensor, config: dict) -> DiffusionTrainer:
    """Trains a diffusion model."""
    trainer = DiffusionTrainer(model, config)
    
    training_history = trainer.train(train_data, val_data, config['epochs'])
    
    trainer.plot_training_curves('.research/iteration1/images/')
    
    model_name = model.model_type.replace('-', '_').lower()
    trainer.save_model(f'models/{model_name}_checkpoint.pth')
    
    return trainer


if __name__ == "__main__":
    from preprocess import create_datasets
    
    config = {
        'image_size': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'train_samples': 120,
        'val_samples': 60,
        'batch_size': 8,
        'epochs': 5,
        'timesteps': 1000,
        'learning_rate': 1e-4,
        'cost_reg_lambda': 0.01,
        'balance_loss_lambda': 0.01
    }
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_type = "Test"
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
        
        def forward(self, x, t):
            return self.conv(x), {'cost': 0.0, 'balance': 0.0}, []
    
    train_data, _, val_data, _ = create_datasets(config)
    model = DummyModel().to(config['device'])
    
    trainer = train_model(model, train_data, val_data, config)
    print("Training module test completed successfully!")
