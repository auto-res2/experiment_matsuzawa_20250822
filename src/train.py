"""
Training script for SEEDS experiments
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.seeds_config import config
from src.models import ImagePtheta, ImageSurrogate, SequencePtheta, SequenceSurrogate
from src.diffusion_utils import BetaSchedule, forward_corrupt, E_site_from_logits, heteroscedastic_loss
from src.preprocess import load_datasets, create_dataloaders

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_ptheta_image(model: ImagePtheta, loader: DataLoader, epochs: int = 5, 
                      lr: float = 1e-3, device: str = 'cpu'):
    """Train image denoising model"""
    print("Training image p_theta model...")
    
    sched = BetaSchedule(config.K, config.beta_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, x0 in enumerate(pbar):
            x0 = x0.to(device)
            batch_size = x0.size(0)
            
            t = torch.rand(batch_size, device=device)
            
            xt = forward_corrupt(x0, t, config.K, sched)
            
            logits = model(xt, t)
            
            loss = F.cross_entropy(
                logits.view(-1, config.K), 
                x0.view(-1).long()
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    return losses

def train_ptheta_sequence(model: SequencePtheta, loader: DataLoader, epochs: int = 5, 
                         lr: float = 1e-3, device: str = 'cpu'):
    """Train sequence denoising model"""
    print("Training sequence p_theta model...")
    
    sched = BetaSchedule(config.K, config.beta_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, x0 in enumerate(pbar):
            x0 = x0.to(device)
            batch_size = x0.size(0)
            
            t = torch.rand(batch_size, device=device)
            
            xt = forward_corrupt(x0, t, config.K, sched)
            
            logits = model(xt, t)
            
            loss = F.cross_entropy(
                logits.view(-1, config.K), 
                x0.view(-1).long()
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    return losses

def generate_surrogate_data(ptheta_model, loader: DataLoader, n_samples: int = 1000, 
                           device: str = 'cpu'):
    """Generate training data for surrogate model"""
    print("Generating surrogate training data...")
    
    sched = BetaSchedule(config.K, config.beta_params)
    ptheta_model.eval()
    
    xs, ts, Es = [], [], []
    
    with torch.no_grad():
        samples_collected = 0
        for x0 in loader:
            if samples_collected >= n_samples:
                break
                
            x0 = x0.to(device)
            batch_size = x0.size(0)
            
            t = torch.rand(batch_size, device=device)
            
            xt = forward_corrupt(x0, t, config.K, sched)
            
            logits = ptheta_model(xt, t)
            
            E = E_site_from_logits(logits, xt, t, sched, config.K)
            
            xs.append(xt.cpu())
            ts.append(t.cpu())
            Es.append(E.cpu())
            
            samples_collected += batch_size
    
    xs = torch.cat(xs, dim=0)[:n_samples]
    ts = torch.cat(ts, dim=0)[:n_samples]
    Es = torch.cat(Es, dim=0)[:n_samples]
    
    print(f"Generated {len(xs)} surrogate training samples")
    return xs, ts, Es

def train_surrogate_image(model: ImageSurrogate, xs: torch.Tensor, ts: torch.Tensor, 
                         Es: torch.Tensor, epochs: int = 3, lr: float = 1e-3, 
                         device: str = 'cpu'):
    """Train image surrogate model"""
    print("Training image surrogate model...")
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(xs, ts, Es)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Surrogate Epoch {epoch+1}/{epochs}")
        for xt, t, E in pbar:
            xt, t, E = xt.to(device), t.to(device), E.to(device)
            
            mean, sigma = model(xt, t)
            loss = heteroscedastic_loss(mean, sigma, E)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Surrogate Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    return losses

def train_surrogate_sequence(model: SequenceSurrogate, xs: torch.Tensor, ts: torch.Tensor, 
                            Es: torch.Tensor, epochs: int = 3, lr: float = 1e-3, 
                            device: str = 'cpu'):
    """Train sequence surrogate model"""
    print("Training sequence surrogate model...")
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(xs, ts, Es)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Surrogate Epoch {epoch+1}/{epochs}")
        for xt, t, E in pbar:
            xt, t, E = xt.to(device), t.to(device), E.to(device)
            
            mean, sigma = model(xt, t)
            loss = heteroscedastic_loss(mean, sigma, E)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Surrogate Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    return losses

def calibrate_kappa(surrogate_model, xs: torch.Tensor, ts: torch.Tensor, 
                   Es: torch.Tensor, delta: float = 0.01, device: str = 'cpu'):
    """Calibrate kappa for conformal prediction"""
    print(f"Calibrating kappa for delta={delta}...")
    
    surrogate_model.eval()
    surrogate_model.to(device)
    
    residuals = []
    
    with torch.no_grad():
        dataset = torch.utils.data.TensorDataset(xs, ts, Es)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        
        for xt, t, E_true in loader:
            xt, t, E_true = xt.to(device), t.to(device), E_true.to(device)
            
            mean, sigma = surrogate_model(xt, t)
            residual = ((E_true - mean) / (sigma + 1e-8)).cpu().numpy()
            residuals.append(residual.flatten())
    
    all_residuals = np.concatenate(residuals)
    
    quantile = 1.0 - delta
    kappa = np.quantile(np.abs(all_residuals), quantile)
    
    print(f"Calibrated kappa: {kappa:.4f}")
    return kappa

def save_models(models_dict: dict, save_dir: str):
    """Save trained models"""
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in models_dict.items():
        path = os.path.join(save_dir, f"{name}.pt")
        torch.save(model.state_dict(), path)
        print(f"Saved {name} to {path}")

def main():
    """Main training function"""
    set_seed(config.seed)
    
    device = config.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    datasets = load_datasets()
    loaders = create_dataloaders(datasets)
    
    image_ptheta = ImagePtheta(config.K)
    image_surrogate = ImageSurrogate(config.K)
    seq_ptheta = SequencePtheta(config.K)
    seq_surrogate = SequenceSurrogate(config.K)
    
    print("=== Training p_theta models ===")
    image_losses = train_ptheta_image(
        image_ptheta, loaders['image_train'], 
        epochs=config.epochs, lr=config.learning_rate, device=device
    )
    
    seq_losses = train_ptheta_sequence(
        seq_ptheta, loaders['seq_train'], 
        epochs=config.epochs, lr=config.learning_rate, device=device
    )
    
    print("=== Generating surrogate data ===")
    image_xs, image_ts, image_Es = generate_surrogate_data(
        image_ptheta, loaders['image_train'], n_samples=500, device=device
    )
    
    seq_xs, seq_ts, seq_Es = generate_surrogate_data(
        seq_ptheta, loaders['seq_train'], n_samples=500, device=device
    )
    
    print("=== Training surrogate models ===")
    image_surr_losses = train_surrogate_image(
        image_surrogate, image_xs, image_ts, image_Es,
        epochs=3, lr=config.learning_rate, device=device
    )
    
    seq_surr_losses = train_surrogate_sequence(
        seq_surrogate, seq_xs, seq_ts, seq_Es,
        epochs=3, lr=config.learning_rate, device=device
    )
    
    print("=== Calibrating kappa ===")
    image_kappa = calibrate_kappa(
        image_surrogate, image_xs, image_ts, image_Es, 
        delta=config.delta, device=device
    )
    
    seq_kappa = calibrate_kappa(
        seq_surrogate, seq_xs, seq_ts, seq_Es, 
        delta=config.delta, device=device
    )
    
    models_dict = {
        'image_ptheta': image_ptheta,
        'image_surrogate': image_surrogate,
        'seq_ptheta': seq_ptheta,
        'seq_surrogate': seq_surrogate
    }
    
    save_models(models_dict, config.models_dir)
    
    kappa_dict = {
        'image_kappa': image_kappa,
        'seq_kappa': seq_kappa
    }
    torch.save(kappa_dict, os.path.join(config.models_dir, 'kappa_values.pt'))
    
    print("Training completed successfully!")
    return {
        'image_losses': image_losses,
        'seq_losses': seq_losses,
        'image_surr_losses': image_surr_losses,
        'seq_surr_losses': seq_surr_losses,
        'image_kappa': image_kappa,
        'seq_kappa': seq_kappa
    }

if __name__ == "__main__":
    results = main()
