import torch
import torch.nn.functional as F
import time
from tqdm import tqdm

def train_one_epoch(model, train_loader, optimizer, scheduler, device, config):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss, total_correct, total_samples, total_density = 0.0, 0.0, 0.0, 0.0
    start_time = time.time()
    
    pbar = tqdm(train_loader, desc=f"Epoch {config['epoch']}", leave=False)
    for i, (img_seq, label) in enumerate(pbar):
        img_seq, label = img_seq.to(device, non_blocking=True), label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        output_seq = model(img_seq) 
        # Loss is based on the mean membrane potential of the output layer over time
        output_potential = output_seq.mean(dim=0)
        loss = F.cross_entropy(output_potential, label)
        
        avg_density = 1.0
        if config['model_type'] == 'FG-OTTT':
            avg_density = model.get_avg_density()
            sparsity_loss = (avg_density - model.s_target)**2
            loss += config['lambda_sparsity'] * sparsity_loss
            total_density += avg_density
        
        loss.backward()

        # For HS-OTTT, sparsify the gradient *after* it has been calculated
        if config['model_type'] == 'HS-OTTT':
            avg_density_batch = 0
            num_params = 0
            for p in model.parameters():
                if p.grad is not None and p.requires_grad:
                    mask = torch.rand_like(p.grad) < model.s_target
                    p.grad.mul_(mask)
                    avg_density_batch += mask.float().sum()
                    num_params += p.numel()
            avg_density = (avg_density_batch / num_params).item() if num_params > 0 else 0.0
            total_density += avg_density

        optimizer.step()
        
        total_loss += loss.item()
        total_correct += (output_potential.argmax(dim=1) == label).sum().item()
        total_samples += label.size(0)
        pbar.set_postfix(loss=loss.item())

    scheduler.step()
    
    # This is a highly simplified approximation of backward FLOPs
    vgg11_cifar_gflops = 0.5 
    bwd_gflops_approx = 2 * vgg11_cifar_gflops * len(pbar) * (total_density / len(pbar) if config['model_type'] in ['FG-OTTT', 'HS-OTTT'] else 1.0)

    epoch_time = time.time() - start_time
    epoch_loss = total_loss / len(pbar)
    epoch_acc = total_correct / total_samples
    epoch_density = total_density / len(pbar) if config['model_type'] in ['FG-OTTT', 'HS-OTTT'] else 1.0

    return epoch_loss, epoch_acc, epoch_time, epoch_density, bwd_gflops_approx
