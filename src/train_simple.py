import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
import os
from .preprocess import get_dataloader

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MinimalUNet(nn.Module):
    def __init__(self, dim=32, channels=3, out_dim=None):
        super().__init__()
        self.channels = channels
        self.out_dim = out_dim if out_dim is not None else channels
        
        time_dim = dim * 2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim*2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim*2, dim*4, 3, padding=1),
            nn.SiLU()
        )
        
        self.time_proj = nn.Linear(time_dim, dim*4)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(dim*4, dim*2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim*2, dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim, self.out_dim, 3, padding=1)
        )
        
    def forward(self, x, time):
        t = self.time_mlp(time.view(-1))
        
        x = self.encoder(x)
        
        x = x + self.time_proj(t)[:, :, None, None]
        
        x = self.decoder(x)
        
        return x

class ICNNUNet(MinimalUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, out_dim=1)
        self.final_act = nn.Softplus()

    def forward(self, x, t):
        out = super().forward(x, t)
        return self.final_act(out).sum(dim=[1, 2, 3])

class InverterUNet(MinimalUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def train_aofm(config):
    print('[INFO] Setting up AOFM training...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')
    
    os.makedirs(config.get('output_dir', './models'), exist_ok=True)
    
    icnn = ICNNUNet(dim=32, channels=3).to(device)
    inverter = InverterUNet(dim=32, channels=3).to(device)
    
    print(f'[INFO] ICNN Parameters: {sum(p.numel() for p in icnn.parameters())/1e6:.2f}M')
    print(f'[INFO] Inverter Parameters: {sum(p.numel() for p in inverter.parameters())/1e6:.2f}M')
    
    opt_icnn = optim.AdamW(icnn.parameters(), lr=config.get('lr', 1e-4), betas=(0.9, 0.999))
    opt_inv = optim.AdamW(inverter.parameters(), lr=config.get('lr', 1e-4), betas=(0.9, 0.999))
    
    num_iterations = config.get('num_iterations', 50)  # Very short for testing
    warmup_steps = config.get('warmup_steps', 5)
    
    sched_icnn = get_cosine_schedule_with_warmup(opt_icnn, warmup_steps, num_iterations)
    sched_inv = get_cosine_schedule_with_warmup(opt_inv, warmup_steps, num_iterations)
    
    dataloader = get_dataloader(
        batch_size=config.get('batch_size', 8),  # Very small for testing
        num_workers=config.get('num_workers', 2)
    )
    data_iter = iter(dataloader)
    
    losses = {'total': [], 'ofm': [], 'inv': []}
    
    print('[INFO] Starting training loop...')
    pbar = tqdm(range(num_iterations))
    
    for step in pbar:
        try:
            x0, _ = next(data_iter)
        except (OSError, StopIteration):
            data_iter = iter(dataloader)
            x0, _ = next(data_iter)
        
        x0 = x0.to(device)
        opt_icnn.zero_grad()
        opt_inv.zero_grad()

        x1 = torch.randn_like(x0)
        t = torch.rand(x0.size(0), 1, 1, 1, device=device)
        xt = (1 - t) * x0 + t * x1
        
        inverted_x0_hat = inverter(xt, t)
        inverted_x0_hat.requires_grad_(True)
        potential_at_inverted_sum = icnn(inverted_x0_hat, t).sum()
        grad_potential = torch.autograd.grad(potential_at_inverted_sum, inverted_x0_hat, create_graph=True)[0]
        kkt_residual = t * grad_potential - xt + (1 - t) * inverted_x0_hat
        loss_inv = torch.mean(kkt_residual.view(x0.size(0), -1).pow(2).sum(dim=1))

        xt.requires_grad_(True)
        potential_at_xt_sum = icnn(xt, t).sum()
        grad_potential_xt = torch.autograd.grad(potential_at_xt_sum, xt, create_graph=True)[0]
        
        v_ofm = x1 - x0
        loss_ofm = torch.mean((grad_potential_xt - v_ofm).pow(2))

        total_loss = loss_ofm + config.get('lambda_inv', 1.0) * loss_inv
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(icnn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(inverter.parameters(), 1.0)
        
        opt_icnn.step()
        opt_inv.step()
        sched_icnn.step()
        sched_inv.step()

        losses['total'].append(total_loss.item())
        losses['ofm'].append(loss_ofm.item())
        losses['inv'].append(loss_inv.item())
        pbar.set_description(f'Total: {total_loss.item():.4f} | OFM: {loss_ofm.item():.4f} | Inv: {loss_inv.item():.4f}')

    torch.save(icnn.state_dict(), os.path.join(config.get('output_dir', './models'), 'icnn_final.pth'))
    torch.save(inverter.state_dict(), os.path.join(config.get('output_dir', './models'), 'inverter_final.pth'))
    
    return {'icnn': icnn, 'inverter': inverter}, losses
