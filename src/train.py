import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import higher
from copy import deepcopy

# --- 1. Helper Functions ---
def accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def velo_loss_fn(losses_over_trajectory, initial_loss, r, alpha=10.0):
    epsilon_task = r * initial_loss.detach()
    total_velo_loss = torch.tensor(0.0, device=losses_over_trajectory[0].device)
    for loss_k in losses_over_trajectory:
        total_velo_loss += torch.sigmoid(alpha * (loss_k - epsilon_task))
    return total_velo_loss

# --- 2. Model Architecture (Conv-4) ---
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, out_features=5, hidden_size=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hidden_size),
            conv_block(hidden_size, hidden_size),
            conv_block(hidden_size, hidden_size),
            conv_block(hidden_size, hidden_size)
        )
        self.classifier = nn.Linear(hidden_size * 5 * 5, out_features)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --- 3. Meta-Learner Model ---
class MetaLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ConvNet(out_features=config['experiment']['n_way'])
        self.inner_lr = config['training']['inner_lr']
        self.algorithm = config['experiment']['algorithm']

        if self.algorithm == 'meta_sgd':
            self.inner_lrs = nn.ParameterDict({
                name.replace('.', '_'): nn.Parameter(torch.full_like(p, self.inner_lr))
                for name, p in self.model.named_parameters()
            })

        if self.algorithm == 'veloml':
            self.r = nn.Parameter(torch.tensor(0.5)) # Meta-learned relative threshold 'r'

    def get_inner_optim(self, params):
        if self.algorithm == 'meta_sgd':
            return optim.SGD(params, lr=self.inner_lr) # Placeholder LR
        else:
            return optim.SGD(params, lr=self.inner_lr)

    def forward(self, support_x, support_y, query_x, query_y, is_train=True):
        if self.algorithm == 'protonet':
            return self.protonet_forward(support_x, support_y, query_x, query_y)
        else:
            return self.optimization_based_forward(support_x, support_y, query_x, query_y, is_train)

    def protonet_forward(self, support_x, support_y, query_x, query_y):
        support_embeddings = self.model.encoder(support_x).view(support_x.size(0), -1)
        query_embeddings = self.model.encoder(query_x).view(query_x.size(0), -1)

        prototypes = []
        for c in range(self.config['experiment']['n_way']):
            class_mask = (support_y == c)
            prototypes.append(support_embeddings[class_mask].mean(dim=0))
        prototypes = torch.stack(prototypes)

        dists = torch.cdist(query_embeddings, prototypes)
        logits = -dists
        loss = F.cross_entropy(logits, query_y)
        acc = accuracy(logits, query_y)
        return loss, acc

    def optimization_based_forward(self, support_x, support_y, query_x, query_y, is_train):
        inner_optimizer = self.get_inner_optim(self.model.parameters())
        num_steps = self.config['veloml']['k_max'] if self.algorithm == 'veloml' and is_train else self.config['training']['num_inner_steps']
        
        query_losses = []
        query_accuracies = []
        support_losses_traj = []

        with higher.innerloop_ctx(self.model, inner_optimizer, copy_initial_weights=is_train, track_higher_grads=is_train) as (fmodel, diffopt):
            initial_support_loss = F.cross_entropy(fmodel(support_x), support_y)
            support_losses_traj.append(initial_support_loss)

            with torch.no_grad():
                query_logits = fmodel(query_x)
                query_losses.append(F.cross_entropy(query_logits, query_y).item())
                query_accuracies.append(accuracy(query_logits, query_y).item())

            for k in range(num_steps):
                if is_train:
                    is_second_order = self.algorithm == 'maml'
                    is_hybrid_step = self.algorithm == 'veloml' and k < self.config['veloml']['k_hybrid']
                    track_grads = is_second_order or is_hybrid_step
                    if hasattr(fmodel, 'track_higher_grads'): fmodel.track_higher_grads = track_grads
                
                support_loss = F.cross_entropy(fmodel(support_x), support_y)
                
                if self.algorithm == 'meta_sgd' and is_train:
                    grads = torch.autograd.grad(support_loss, fmodel.parameters(time=k), create_graph=track_grads)
                    fmodel.update_params([p - self.inner_lrs[name.replace('.', '_')] * g for (name, p), g in zip(fmodel.named_parameters(time=k), grads)])
                else:
                    diffopt.step(support_loss)
                
                if self.algorithm == 'veloml' and is_train:
                    support_losses_traj.append(F.cross_entropy(fmodel(support_x), support_y))

                with torch.no_grad():
                    query_logits = fmodel(query_x)
                    query_losses.append(F.cross_entropy(query_logits, query_y).item())
                    query_accuracies.append(accuracy(query_logits, query_y).item())
            
            if is_train:
                if self.algorithm == 'veloml':
                    meta_loss = velo_loss_fn(support_losses_traj[1:], initial_support_loss, self.r)
                else:
                    final_query_logits = fmodel(query_x)
                    meta_loss = F.cross_entropy(final_query_logits, query_y)
            else:
                meta_loss = torch.tensor(0.0)

        return meta_loss, query_accuracies

    def run_adaptive_inference(self, support_x, support_y, query_x, query_y):
        if self.algorithm != 'veloml':
            raise ValueError("Adaptive inference is only for VeloML.")

        adapted_model = deepcopy(self.model)
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        initial_loss = F.cross_entropy(adapted_model(support_x), support_y)
        threshold = self.r.item() * initial_loss.item()
        
        num_steps = 0
        k_max_inference = 50
        for k in range(k_max_inference):
            num_steps += 1
            loss = F.cross_entropy(adapted_model(support_x), support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss.item() <= threshold:
                break
        
        adapted_model.eval()
        with torch.no_grad():
            query_logits = adapted_model(query_x)
            acc = accuracy(query_logits, query_y).item()

        return acc, num_steps

# --- 4. Training Loop ---
def train_epoch(model, dataloader, optimizer, config, device):
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = config['training']['num_batches']
    
    pbar = tqdm(dataloader, total=num_batches, desc="Meta-Training Epoch")
    for i, batch in enumerate(pbar):
        if i >= num_batches:
            break

        support_x, support_y = batch['train']
        query_x, query_y = batch['test']
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)
        
        support_x = support_x.view(-1, *support_x.shape[2:])
        support_y = support_y.view(-1)
        query_x = query_x.view(-1, *query_x.shape[2:])
        query_y = query_y.view(-1)

        optimizer.zero_grad()
        meta_loss, query_accuracies = model(support_x, support_y, query_x, query_y, is_train=True)
        meta_loss.backward()
        optimizer.step()

        total_loss += meta_loss.item()
        final_acc = query_accuracies[-1] if isinstance(query_accuracies, list) else query_accuracies
        total_acc += final_acc

        pbar.set_postfix(loss=f"{meta_loss.item():.4f}", acc=f"{final_acc:.4f}")

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    if config['experiment']['algorithm'] == 'veloml':
        print(f"Meta-learned 'r': {model.r.item():.4f}")
        
    return avg_loss, avg_acc
