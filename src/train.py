import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from copy import deepcopy
import math
import os

# --- Model Definition ---
class MoCoResNet(nn.Module):
    def __init__(self, config):
        super(MoCoResNet, self).__init__()
        self.config = config

        # Create encoders
        base_encoder = models.resnet18(weights=None, num_classes=config['MOCO_DIM'])
        
        # Modify ResNet for CIFAR-100 (smaller images)
        base_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_encoder.maxpool = nn.Identity()
        
        dim_mlp = base_encoder.fc.in_features
        # Add projector head
        base_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), base_encoder.fc)

        self.encoder_q = base_encoder
        self.encoder_k = deepcopy(base_encoder)

        self._init_encoder_k()

        # Create the queue
        self.register_buffer("queue", torch.randn(config['MOCO_DIM'], config['MOCO_K']))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        m = self.config['MOCO_M']
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if self.config['MOCO_K'] % batch_size != 0:
             # This case should be handled in data loader with drop_last=True
             return
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.config['MOCO_K']
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k=None, is_pretrain=True):
        if not is_pretrain:
            # For linear evaluation, only use the query encoder for feature extraction
            return self.encoder_q(im_q)
        
        # Compute query features
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # Compute key features with momentum encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        return q, k

# --- Loss Functions ---
def moco_loss(q, k, queue, temp):
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= temp
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)

def moco_mts_loss(model, im_q, q, k, queue, config):
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

    with torch.no_grad():
        q_momentum = F.normalize(model.encoder_k(im_q), dim=1)

    hard_vals, hard_indices = torch.topk(l_neg, config['MTS_K_HARD'], dim=1)
    hard_k_neg = queue.clone().detach().T[hard_indices]
    s_historical = torch.einsum('nc,nkc->nk', [q_momentum, hard_k_neg])
    s_current = hard_vals
    mts = F.relu(s_current - s_historical)
    sim_adjusted = s_current - config['MTS_LAMBDA'] * mts
    l_neg.scatter_(1, hard_indices, sim_adjusted)

    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= config['MOCO_T']
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)

def moco_dcl_loss(q, k, queue, config):
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
    debias_term = math.log(1.0 - config['DCL_TAU_PLUS'])
    l_neg += debias_term
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= config['MOCO_T']
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)

def moco_srns_loss(q, k, queue, config):
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    l_neg_all = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

    with torch.no_grad():
        sim_matrix_T = torch.einsum('kc,nc->kn', [queue.clone().detach().T, q])
        score_var = torch.var(sim_matrix_T, dim=1)
        mask = (score_var >= config['SRNS_VAR_THRESH']).float().unsqueeze(0)

    l_neg = torch.where(mask > 0, l_neg_all, torch.tensor(-1e9).to(q.device))
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= config['MOCO_T']
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)

# --- Training Loops ---
def pretrain_one_epoch(model, dataloader, optimizer, epoch, config, method):
    model.train()
    total_loss = 0
    for i, (images, _) in enumerate(dataloader):
        im_q, im_k = images[0].to(config['DEVICE']), images[1].to(config['DEVICE'])
        q, k = model(im_q, im_k)

        if method == 'moco':
            loss = moco_loss(q, k, model.queue, config['MOCO_T'])
        elif method == 'mts':
            loss = moco_mts_loss(model, im_q, q, k, model.queue, config)
        elif method == 'dcl':
            loss = moco_dcl_loss(q, k, model.queue, config)
        elif method == 'srns':
            loss = moco_srns_loss(q, k, model.queue, config)
        else:
            raise ValueError(f"Unknown pre-training method: {method}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        model._dequeue_and_enqueue(k)

    avg_loss = total_loss / len(dataloader)
    print(f"[{method.upper()}] Pre-train Epoch {epoch+1}/{config['PRETRAIN_EPOCHS']}, Loss: {avg_loss:.4f}")
    return avg_loss

def train_linear_one_epoch(backbone, classifier, dataloader, criterion, optimizer, config):
    backbone.eval()
    classifier.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for images, labels in dataloader:
        images, labels = images.to(config['DEVICE']), labels.to(config['DEVICE'])
        with torch.no_grad():
            features = backbone(images, is_pretrain=False)
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy

def run_pretraining(config, pretrain_loader, method):
    print(f'\n--- Starting Pre-training for: {method.upper()} ---')
    model = MoCoResNet(config).to(config['DEVICE'])
    optimizer = optim.SGD(model.parameters(), lr=config['PRETRAIN_LR'], momentum=0.9, weight_decay=1e-4)
    
    loss_history = []
    for epoch in range(config['PRETRAIN_EPOCHS']):
        loss = pretrain_one_epoch(model, pretrain_loader, optimizer, epoch, config, method)
        loss_history.append(loss)
    
    checkpoint_path = os.path.join(config['MODELS_DIR'], f'{method}_pretrained.pth')
    torch.save({'model': model.state_dict()}, checkpoint_path)
    print(f'Saved {method.upper()} pre-trained model to {checkpoint_path}')
    return checkpoint_path, loss_history

def run_linear_training(config, train_loader, test_loader, pretrained_path, method, evaluate_fn):
    print(f'\n--- Starting Linear Training for: {method.upper()} ---')
    
    if method == 'supervised':
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        model = MoCoResNet(config)
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        backbone = model.encoder_q
        num_ftrs = backbone.fc[0].in_features
        backbone.fc = nn.Identity()

    backbone = backbone.to(config['DEVICE'])
    for param in backbone.parameters():
        param.requires_grad = False
    
    classifier = nn.Linear(num_ftrs, config['NUM_CLASSES']).to(config['DEVICE'])
    optimizer = optim.Adam(classifier.parameters(), lr=config['LINEAR_LR'])
    criterion = nn.CrossEntropyLoss()
    
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []

    for epoch in range(config['LINEAR_EPOCHS']):
        train_loss, train_acc = train_linear_one_epoch(backbone, classifier, train_loader, criterion, optimizer, config)
        val_loss, val_acc = evaluate_fn(backbone, classifier, test_loader, criterion, config)
        
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        print(f'[{method.upper()}] Linear Eval Epoch {epoch+1}/{config['LINEAR_EPOCHS']} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')

    _, final_test_acc = evaluate_fn(backbone, classifier, test_loader, criterion, config)
    print(f'---> [{method.upper()}] Final Test Accuracy: {final_test_acc:.2f}% <---')
    
    history = {
        'train_loss': train_loss_hist,
        'val_acc': val_acc_hist,
        'final_acc': final_test_acc
    }
    return history
