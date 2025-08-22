import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import time

class GCNEncoder(nn.Module):
    """A two-layer GCN encoder as specified."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class ProjectionHead(nn.Module):
    """A 2-layer MLP projection head as specified."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)

class BaseGCL(nn.Module):
    """Base class for Graph Contrastive Learning models."""
    def __init__(self, encoder, projection_head):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x, edge_index, edge_weight=None):
        return self.encoder(x, edge_index, edge_weight)

    def project(self, z):
        return self.projection_head(z)

    def get_loss(self, h1, h2):
        raise NotImplementedError

class InfoNCE_GCL(BaseGCL):
    """A generic GCL model using the standard InfoNCE loss (for GRACE and GCA baselines)."""
    def __init__(self, encoder, projection_head, temp=0.1):
        super().__init__(encoder, projection_head)
        self.temp = temp

    def get_loss(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)

        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / self.temp
        
        numerator = torch.exp(sim_matrix.diag(batch_size))
        
        neg_mask = torch.ones_like(sim_matrix).fill_diagonal_(0)
        neg_mask[:batch_size, batch_size:] = 0
        neg_mask[batch_size:, :batch_size] = 0
        denominator = torch.exp(sim_matrix) * neg_mask
        
        loss = -torch.log(numerator / denominator.sum(dim=1)[:batch_size])
        return loss.mean()

class ASAR_GCL(BaseGCL):
    """The proposed ASAR model, which can also function as its ablations."""
    def __init__(self, encoder, projection_head, dynamic_lambda=True, dynamic_margin=True,
                 fixed_lambda=1.0, fixed_margin=0.1, alpha=0.1, lambda_0=1.0, 
                 temp=0.1, clip_max=5.0):
        super().__init__(encoder, projection_head)
        self.dynamic_lambda = dynamic_lambda
        self.dynamic_margin = dynamic_margin
        self.fixed_lambda = fixed_lambda
        self.fixed_margin = fixed_margin
        self.alpha = alpha
        self.lambda_0 = lambda_0
        self.temp = temp
        self.clip_max = clip_max

    def get_loss(self, z1, z2):
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)

        batch_size = z1.size(0)
        z = torch.cat([z1_norm, z2_norm], dim=0)
        
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        
        sim_pos = F.cosine_similarity(z1_norm, z2_norm, dim=-1)
        numerator = torch.exp(sim_pos / self.temp)

        neg_mask = torch.ones(batch_size * 2, batch_size * 2, device=z.device).fill_diagonal_(0)
        neg_mask[:batch_size, batch_size:] = 0
        neg_mask[batch_size:, :batch_size] = 0
        sim_matrix_exp = torch.exp(sim_matrix / self.temp)
        denominator = (sim_matrix_exp * neg_mask).sum(dim=1)[:batch_size]
        
        l_align = -torch.log(numerator / denominator).mean()
        
        with torch.no_grad():
            non_diag_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
            non_diag_sims = sim_matrix[non_diag_mask]
            
            if self.dynamic_margin:
                dynamic_margin = self.alpha * torch.std(non_diag_sims)
            else:
                dynamic_margin = self.fixed_margin

            if self.dynamic_lambda:
                lambda_t = self.lambda_0 * torch.clamp(torch.mean(non_diag_sims) + 0.5, min=0.0, max=self.clip_max)
            else:
                lambda_t = self.fixed_lambda

            pos_pair_sims = sim_matrix.diag(batch_size)
            margin_threshold = pos_pair_sims - dynamic_margin
            margin_threshold = torch.cat([margin_threshold, margin_threshold])
            
            is_confusable = sim_matrix > margin_threshold.unsqueeze(1)
            
            is_confusable *= neg_mask.bool()

        repulsion_energies = sim_matrix_exp * is_confusable
        num_active_anchors = max(1, (is_confusable.sum(dim=1) > 0).sum().item())
        l_asar = repulsion_energies.sum() / num_active_anchors
        
        return l_align + lambda_t * l_asar

def drop_features(x, p):
    """Drop features randomly with probability p."""
    mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, mask] = 0
    return x

def drop_edges(edge_index, p):
    """Drop edges randomly with probability p."""
    if p == 0: 
        return edge_index
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > p
    return edge_index[:, mask]

def augment_data(x, edge_index, feat_mask_rate=0.5, edge_drop_rate=0.2):
    """Simple augmentation creating two views of the graph."""
    x1 = drop_features(x, feat_mask_rate)
    edge_index1 = drop_edges(edge_index, edge_drop_rate)
    x2 = drop_features(x, feat_mask_rate)
    edge_index2 = drop_edges(edge_index, edge_drop_rate)
    return x1, edge_index1, x2, edge_index2

def train_pretext(model, loader, optimizer, scheduler, device):
    """Train the model for one epoch using contrastive learning."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        x1, edge_index1, x2, edge_index2 = augment_data(batch.x, batch.edge_index)
        
        z1 = model(x1, edge_index1)
        z2 = model(x2, edge_index2)
        
        h1 = model.project(z1)
        h2 = model.project(z2)
        
        loss = model.get_loss(h1, h2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    if scheduler:
        scheduler.step()
    return total_loss / max(num_batches, 1)

def train_model(model, loader, num_epochs, device, lr=1e-3, weight_decay=1e-5):
    """Train a model for the specified number of epochs."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        loss = train_pretext(model, loader, optimizer, scheduler, device)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    
    end_time = time.time()
    training_time = end_time - start_time
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == 'cuda' else 0
    
    print(f"Training finished. Time: {training_time:.2f}s, Peak Memory: {peak_mem_gb:.3f} GB")
    return training_time, peak_mem_gb
