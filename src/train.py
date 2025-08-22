import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.linalg import eigh
from torch.utils.data import DataLoader
import copy


class SimpleNet(nn.Module):
    """Simple neural network for continual learning experiments."""
    
    def __init__(self, input_size, hidden_size=256, num_classes=10):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class SOCATrainer:
    """Simplified SOCA trainer for memory efficiency."""
    
    def __init__(self, model, device, k_global=5, alpha=0.9, reg_lambda=0.001):
        self.model = model
        self.device = device
        self.k_global = k_global
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        
        self.task_count = 0
        self.theta_star = None
        
    def train_task(self, train_loader, epochs=3, lr=0.001):
        """Train model on a single task with ultra-lightweight SOCA regularization."""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        if self.theta_star is None:
            self.theta_star = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                task_loss = criterion(output, target)
                
                reg_loss = 0.0
                if self.task_count > 0:
                    for name, param in self.model.named_parameters():
                        if name in self.theta_star:
                            param_diff = param - self.theta_star[name].to(self.device)
                            reg_loss += self.reg_lambda * torch.norm(param_diff) ** 2
                
                total_loss_tensor = task_loss + reg_loss
                total_loss_tensor.backward()
                optimizer.step()
                
                total_loss += total_loss_tensor.item()
        
        self.theta_star = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        self.task_count += 1
        
        return total_loss / len(train_loader)


class BaselineTrainer:
    """Baseline trainer for comparison methods."""
    
    def __init__(self, model, device, method='finetune'):
        self.model = model
        self.device = device
        self.method = method
        self.replay_buffer = []
        self.buffer_size = 1000
        
    def train_task(self, train_loader, epochs=5, lr=0.001):
        """Train model using baseline method."""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        if self.method == 'experience_replay':
            for data, target in train_loader:
                for i in range(data.size(0)):
                    if len(self.replay_buffer) < self.buffer_size:
                        self.replay_buffer.append((data[i], target[i]))
                    else:
                        idx = np.random.randint(0, self.buffer_size)
                        self.replay_buffer[idx] = (data[i], target[i])
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                
                if self.method == 'experience_replay' and len(self.replay_buffer) > 0:
                    replay_data = []
                    replay_targets = []
                    
                    n_replay = min(32, len(self.replay_buffer))
                    replay_samples = np.random.choice(len(self.replay_buffer), n_replay, replace=False)
                    
                    for idx in replay_samples:
                        replay_data.append(self.replay_buffer[idx][0])
                        replay_targets.append(self.replay_buffer[idx][1])
                    
                    if replay_data:
                        replay_data = torch.stack(replay_data).to(self.device)
                        replay_targets = torch.stack(replay_targets).to(self.device)
                        
                        replay_output = self.model(replay_data)
                        replay_loss = criterion(replay_output, replay_targets)
                        loss = loss + 0.5 * replay_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return total_loss / len(train_loader)
