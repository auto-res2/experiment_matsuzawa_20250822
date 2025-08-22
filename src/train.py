import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict

# ===============================================
# Models
# ===============================================

class MockBackbone(nn.Module):
    """A simple CNN backbone."""
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv_block(x)

class MockTaskHead(nn.Module):
    """A simple task-specific head."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(self.upsample(x))

class MTLModel(nn.Module):
    """Combines the backbone and task heads."""
    def __init__(self, tasks=['segmentation', 'depth']):
        super().__init__()
        self.backbone = MockBackbone()
        task_out_channels = {'segmentation': 13, 'depth': 1}
        self.heads = nn.ModuleDict({
            task: MockTaskHead(32, task_out_channels[task]) for task in tasks
        })

    def forward(self, x):
        shared_features = self.backbone(x)
        return {task: self.heads[task](shared_features) for task in self.heads}

# ===============================================
# Meta-Grad Implementation
# ===============================================

class MetaGradPolicy(nn.Module):
    """Learns low-rank gating masks for gradients."""
    def __init__(self, model, tasks, rank):
        super().__init__()
        self.tasks = tasks
        self.shared_params = [p for p in model.backbone.parameters() if p.requires_grad]
        self.num_shared_params = sum(p.numel() for p in self.shared_params)

        self.A = nn.ParameterDict({task: nn.Parameter(torch.randn(self.num_shared_params, rank)) for task in tasks})
        self.B = nn.ParameterDict({task: nn.Parameter(torch.randn(rank, 1)) for task in tasks})

    def forward(self, per_task_grads):
        """Aggregates gradients using the learned policy."""
        logits = {}
        for task in self.tasks:
            logits[task] = (self.A[task] @ self.B[task]).squeeze(-1)
        
        stacked_logits = torch.stack([logits[task] for task in self.tasks], dim=1)
        gating_weights = torch.sigmoid(stacked_logits)

        stacked_grads = torch.stack([per_task_grads[task] for task in self.tasks], dim=1)
        
        aggregated_grad = torch.sum(gating_weights * stacked_grads, dim=1)
        return aggregated_grad

# ===============================================
# Gradient Utilities & Baseline Methods
# ===============================================

def get_shared_params_grad_dict(model, task_losses, lambdas):
    """Computes and flattens gradients for shared parameters for each task."""
    shared_params = [p for p in model.backbone.parameters() if p.requires_grad]
    per_task_grads = {}
    for i, task in enumerate(task_losses):
        model.zero_grad()
        task_losses[task].backward(retain_graph=True)
        
        flat_grad = torch.cat([p.grad.detach().clone().view(-1) for p in shared_params])
        per_task_grads[task] = flat_grad * lambdas[i]
    model.zero_grad()
    return per_task_grads

def set_shared_params_grad(model, flat_grad):
    """Applies a flattened gradient vector to the model's shared parameters."""
    shared_params = [p for p in model.backbone.parameters() if p.requires_grad]
    offset = 0
    for p in shared_params:
        numel = p.numel()
        p.grad = flat_grad[offset:offset + numel].view_as(p).clone()
        offset += numel

def pcgrad_update(per_task_grads):
    """PCGrad implementation."""
    grads = list(per_task_grads.values())
    num_tasks = len(grads)
    
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            g_i = grads[i]
            g_j = grads[j]
            dot_product = torch.dot(g_i, g_j)
            if dot_product < 0:
                g_i -= (dot_product / torch.dot(g_j, g_j)) * g_j
                g_j -= (dot_product / torch.dot(g_i, g_i)) * g_i
    return sum(grads)

def cagrad_update(per_task_grads):
    """CAGrad implementation (simplified solver)."""
    grads = torch.stack(list(per_task_grads.values()))
    num_tasks = grads.shape[0]
    g_avg = grads.mean(0)
    c = 0.5
    
    if torch.all(torch.matmul(grads, g_avg) >= 0):
        return g_avg * num_tasks

    GG = torch.matmul(grads, grads.t())
    
    def obj(alpha):
        return (c * torch.norm(g_avg) + torch.sqrt(torch.matmul(torch.matmul(alpha, GG), alpha) + 1e-8)).cpu().numpy()
    
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.})
    bounds = tuple((0, 1) for _ in range(num_tasks))
    alpha0 = np.ones(num_tasks) / num_tasks

    sol = minimize(obj, alpha0, method='SLSQP', bounds=bounds, constraints=cons)
    alpha = torch.from_numpy(sol.x).float().to(grads.device)
    return torch.matmul(alpha, grads)

# ===============================================
# Training Loop
# ===============================================

def train_epoch(model, optimizers, policy, train_loader, method, lambdas, device, tasks):
    model.train()
    if policy: policy.train()

    total_losses = defaultdict(float)
    total_meta_loss = 0.0
    
    optimizer_model = optimizers['model']
    if 'policy' in optimizers: optimizer_policy = optimizers['policy']

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        for k in targets: targets[k] = targets[k].to(device)

        outputs = model(data)
        task_losses = {
            'segmentation': nn.CrossEntropyLoss()(outputs['segmentation'], targets['segmentation']),
            'depth': nn.MSELoss()(outputs['depth'], targets['depth'])
        }
        
        optimizer_model.zero_grad()
        if method == 'MetaGrad': optimizer_policy.zero_grad()
        
        if method != 'MetaGrad':
            per_task_grads = get_shared_params_grad_dict(model, task_losses, lambdas)
            
            if method == 'Uniform':
                agg_grad = sum(per_task_grads.values())
            elif method == 'PCGrad':
                agg_grad = pcgrad_update(per_task_grads)
            elif method == 'CAGrad':
                agg_grad = cagrad_update(per_task_grads)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            set_shared_params_grad(model, agg_grad)
            total_loss_for_heads = sum(lambdas[i] * l for i, l in enumerate(task_losses.values()))
            total_loss_for_heads.backward()
            optimizer_model.step()
        else: # MetaGrad
            shared_params = [p for p in model.backbone.parameters() if p.requires_grad]
            per_task_grads_for_policy = {}
            for i, task in enumerate(tasks):
                grads = torch.autograd.grad(lambdas[i] * task_losses[task], shared_params, retain_graph=True)
                per_task_grads_for_policy[task] = torch.cat([g.view(-1) for g in grads])

            aggregated_grad_vec = policy.forward(per_task_grads_for_policy)

            inner_products = [torch.dot(aggregated_grad_vec, per_task_grads_for_policy[task]) for task in tasks]
            meta_loss = -torch.min(torch.stack(inner_products))
            meta_loss.backward()
            optimizer_policy.step()
            total_meta_loss += meta_loss.item()

            set_shared_params_grad(model, aggregated_grad_vec.detach())
            sum(lambdas[i] * l for i, l in enumerate(task_losses.values())).backward()
            optimizer_model.step()

        for task, loss in task_losses.items():
            total_losses[task] += loss.item() * lambdas[tasks.index(task)]

    avg_losses = {task: loss / len(train_loader) for task, loss in total_losses.items()}
    avg_meta_loss = total_meta_loss / len(train_loader) if method == 'MetaGrad' else 0
    return avg_losses, avg_meta_loss
