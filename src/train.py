import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import time
from collections import defaultdict
from tqdm import tqdm

class LGNLayer(nn.Module):
    """A simplified Logic Gate Network layer using smooth activations."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.SiLU()  # Smooth approximation of gate logic
        
    def forward(self, x):
        return self.activation(self.linear(x))

class AJCLGNLayer(LGNLayer):
    """LGN layer with Adaptive Jacobian Control mechanism."""
    def __init__(self, dim):
        super().__init__(dim)
        self.log_lambda = nn.Parameter(torch.tensor(-6.0))
        
        self.register_buffer('deviation_ema', torch.tensor(1.0))
        self.ema_beta = 0.99
        self.controller_gain = 0.01
        self.target_deviation = 1.0
        
    def forward(self, x):
        identity = x
        out = super().forward(x)
        return identity + out
    
    def compute_regularization_loss(self, x):
        """Compute ||J^T J - I||_F^2 using Hutchinson's method."""
        if not self.training:
            return torch.tensor(0.0, device=x.device)
        
        x_reg = x.detach().requires_grad_(True)
        
        v = torch.randn_like(x_reg)
        v = v / torch.norm(v, dim=-1, keepdim=True)
        
        layer_output = super().forward(x_reg)
        
        jvp = torch.autograd.grad(
            layer_output, x_reg, 
            grad_outputs=v, 
            retain_graph=True, 
            create_graph=True
        )[0]
        
        vjp_jvp = torch.autograd.grad(
            layer_output, x_reg, 
            grad_outputs=jvp, 
            retain_graph=False
        )[0]
        
        deviation = vjp_jvp - v
        deviation_sq = (deviation ** 2).sum()
        
        with torch.no_grad():
            current_deviation = torch.sqrt(deviation_sq.detach() + 1e-8)
            self.deviation_ema = (self.ema_beta * self.deviation_ema + 
                                (1 - self.ema_beta) * current_deviation)
            
            error = self.deviation_ema - self.target_deviation
            update = self.controller_gain * torch.clamp(error, -0.1, 0.1)
            self.log_lambda.data = torch.clamp(self.log_lambda.data + update, -10.0, -2.0)
        
        lambda_val = torch.exp(self.log_lambda)
        return lambda_val * deviation_sq

class BaseLGN(nn.Module):
    """Base class for Logic Gate Network models."""
    def __init__(self, depth, num_classes, layer_type, init_fn):
        super().__init__()
        self.depth = depth
        self.feature_dim = 256
        
        self.input_proj = nn.Linear(3 * 32 * 32, self.feature_dim)
        
        self.layers = nn.ModuleList([
            layer_type(self.feature_dim) for _ in range(depth)
        ])
        
        self.output_proj = nn.Linear(self.feature_dim, num_classes)
        
        self.apply(init_fn)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = self.input_proj(x)
        
        self.activations = [x.detach()]
        
        for layer in self.layers:
            x = layer(x)
            self.activations.append(x.detach())
            
        return self.output_proj(x)
    
    def get_regularization_loss(self):
        """Default: no regularization."""
        return torch.tensor(0.0, device=next(self.parameters()).device)

class LGN_Vanilla(BaseLGN):
    """Vanilla LGN with standard Kaiming initialization."""
    def __init__(self, depth, num_classes=100):
        def kaiming_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        super().__init__(depth, num_classes, LGNLayer, kaiming_init)

class LGN_Residual(BaseLGN):
    """LGN with residual initialization (SOTA baseline)."""
    def __init__(self, depth, num_classes=100):
        def kaiming_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        def residual_init(m):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        super().__init__(depth, num_classes, LGNLayer, kaiming_init)
        self.layers.apply(residual_init)

class AJC_LGN(BaseLGN):
    """AJC-LGN with Adaptive Jacobian Control."""
    def __init__(self, depth, num_classes=100):
        def kaiming_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        super().__init__(depth, num_classes, AJCLGNLayer, kaiming_init)
    
    def get_regularization_loss(self):
        """Compute total regularization loss across all layers."""
        if not self.training or not hasattr(self, 'activations'):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        reg_loss = 0.0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, AJCLGNLayer):
                layer_input = self.activations[i]
                reg_loss += layer.compute_regularization_loss(layer_input)
        
        return reg_loss / self.depth  # Average across layers

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        main_loss = loss_fn(output, target)
        reg_loss = model.get_regularization_loss()
        total_loss_batch = main_loss + reg_loss
        
        if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
            print(f'Loss diverged to {total_loss_batch.item()} at batch {batch_idx}')
            return None, None, None  # Signal failure
        
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += main_loss.item()
        total_reg_loss += reg_loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({
            'Loss': f'{main_loss.item():.4f}',
            'Reg': f'{reg_loss.item():.6f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    accuracy = correct / max(total, 1)  # Prevent division by zero
    
    return avg_loss, avg_reg_loss, accuracy

def evaluate_model(model, dataloader, loss_fn, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating", leave=False):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = loss_fn(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / max(total, 1)  # Prevent division by zero
    
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, epochs, lr, weight_decay, device, model_name="model"):
    """Complete training loop with learning rate scheduling."""
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    warmup_epochs = min(5, epochs)
    if epochs > warmup_epochs:
        main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=epochs)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'reg_loss': []
    }
    
    print(f"\nTraining {model_name} for {epochs} epochs...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, reg_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        if train_loss is None:  # Training diverged
            print(f"Training diverged at epoch {epoch+1}")
            return None, False
        
        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['reg_loss'].append(reg_loss)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1:3d}/{epochs} | '
              f'Time: {epoch_time:.1f}s | '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.4f} | '
              f'Reg Loss: {reg_loss:.6f}')
    
    return history, True

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models = {
        'AJC-LGN': AJC_LGN(depth=16),
        'LGN-Residual': LGN_Residual(depth=16),
        'LGN-Vanilla': LGN_Vanilla(depth=16)
    }
    
    for name, model in models.items():
        model = model.to(device)
        print(f"\n{name}:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        x = torch.randn(4, 3, 32, 32).to(device)
        try:
            with torch.no_grad():
                output = model(x)
            print(f"  Forward pass: OK, output shape {output.shape}")
            
            reg_loss = model.get_regularization_loss()
            print(f"  Regularization loss: {reg_loss.item():.6f}")
            
        except Exception as e:
            print(f"  Forward pass: FAILED - {e}")
    
    print("\nModel testing completed!")
