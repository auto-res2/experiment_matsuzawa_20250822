import torch
import torch.nn as nn
from collections import defaultdict
from train import MTLModel

def evaluate_model(model, val_loader, device):
    model.eval()
    total_losses = defaultdict(float)
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            for k in targets: targets[k] = targets[k].to(device)
            
            outputs = model(data)
            total_losses['segmentation'] += nn.CrossEntropyLoss()(outputs['segmentation'], targets['segmentation']).item()
            total_losses['depth'] += nn.MSELoss()(outputs['depth'], targets['depth']).item()
    
    avg_seg_loss = total_losses['segmentation'] / len(val_loader)
    avg_depth_loss = total_losses['depth'] / len(val_loader)
    
    # Negative loss -> higher is better for Pareto plots
    return {
        'segmentation_perf': -avg_seg_loss,
        'depth_perf': -avg_depth_loss
    }
