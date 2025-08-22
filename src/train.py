import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18, vgg16_bn
from collections import defaultdict
from torch.utils.data import DataLoader

# Try to import optional libraries
try:
    from diffusers import DDPMPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: 'diffusers' library not found. PureGen and ULTRA-C+ with DDPM will not be available.")

try:
    from torch.func import vmap, grad
    FUNCTORCH_AVAILABLE = True
except ImportError:
    FUNCTORCH_AVAILABLE = False
    print("Warning: functorch (torch.func) not available. ULTRA-C+ and EPIC will not be available.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 3. Model Definitions
# =============================================================================

def get_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = resnet18(weights=None, num_classes=num_classes)
    elif model_name == 'vgg16_bn':
        model = vgg16_bn(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
    return model.to(DEVICE)

# =============================================================================
# 4. ULTRA-C+ Core Components
# =============================================================================

class TrajectoryTracker:
    def __init__(self, num_samples, window_size=16):
        self.K = window_size
        self.loss_history = torch.zeros(num_samples, self.K).to(DEVICE)
        self.history_counts = torch.zeros(num_samples, dtype=torch.long).to(DEVICE)

    def update_and_get_features(self, losses, indices):
        self.loss_history[indices] = torch.roll(self.loss_history[indices], shifts=-1, dims=1)
        self.loss_history[indices, -1] = losses.detach()
        self.history_counts[indices] = torch.min(self.history_counts[indices] + 1, torch.tensor(self.K, device=DEVICE))
        
        current_history = self.loss_history[indices]
        valid_mask = (self.history_counts[indices] > 4).float()

        variance = torch.var(current_history, dim=1)
        
        x = torch.arange(self.K - 1, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        y = current_history[:, :-1]
        
        x_mean, y_mean = torch.mean(x), torch.mean(y, dim=1, keepdim=True)
        b = torch.sum((x - x_mean) * (y - y_mean), dim=1) / (torch.sum((x - x_mean)**2) + 1e-6)
        a = y_mean.squeeze() - b * x_mean
        
        predicted_kth_loss = a + b * (self.K - 1)
        actual_kth_loss = current_history[:, -1]
        extrapolation_error = (predicted_kth_loss - actual_kth_loss)**2

        variance = (variance - torch.mean(variance)) / (torch.std(variance) + 1e-6)
        extrapolation_error = (extrapolation_error - torch.mean(extrapolation_error)) / (torch.std(extrapolation_error) + 1e-6)

        features = torch.stack([variance, extrapolation_error], dim=1)
        return features * valid_mask.unsqueeze(1)

class ScoringHead(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze()

class CGSManager:
    def __init__(self, num_classes, model, beta=0.995):
        if not FUNCTORCH_AVAILABLE: raise ImportError("CGSManager requires torch.func (functorch).")
        self.num_classes = num_classes
        self.beta = beta
        self.ema_grads = {c: [torch.zeros_like(p.data) for p in model.parameters()] for c in range(num_classes)}

    def update_ema(self, per_sample_grads, labels, clean_mask):
        with torch.no_grad():
            for c in range(self.num_classes):
                class_mask = (labels == c) & clean_mask
                if torch.any(class_mask):
                    class_grads = [p_grad[class_mask].mean(dim=0) for p_grad in per_sample_grads]
                    for i, ema_grad in enumerate(self.ema_grads[c]):
                        ema_grad.mul_(self.beta).add_(class_grads[i], alpha=1-self.beta)
    
    def steer_gradients(self, per_sample_grads, labels, anomaly_scores, score_threshold):
        corrected_grads = [p_grad.clone() for p_grad in per_sample_grads]
        suspect_mask = anomaly_scores > score_threshold
        if not torch.any(suspect_mask): return corrected_grads

        alpha = torch.clamp(2.0 * (anomaly_scores - score_threshold), 0.0, 0.8)
        
        for i in range(len(corrected_grads)):
            for sample_idx in torch.where(suspect_mask)[0]:
                c = labels[sample_idx].item()
                g_suspect = per_sample_grads[i][sample_idx]
                g_clean_ema = self.ema_grads[c][i]
                g_corrected = g_suspect + alpha[sample_idx] * (g_clean_ema - g_suspect)
                corrected_grads[i][sample_idx] = g_corrected
        return corrected_grads

class DDPMPurifier:
    def __init__(self, model_id='google/ddpm-cifar10-32', steps=2):
        if not DIFFUSERS_AVAILABLE: raise ImportError("DDPMPurifier requires the 'diffusers' library.")
        self.steps = steps
        self.pipeline = DDPMPipeline.from_pretrained(model_id).to(DEVICE)
        self.pipeline.scheduler.set_timesteps(num_inference_steps=steps)
        print(f"DDPM Purifier loaded with {steps} steps.")

    @torch.no_grad()
    def purify(self, images):
        images_rescaled = (images * 2) - 1
        purified_images = self.pipeline(batch_size=images.shape[0], output_type="pt", return_dict=False)[0]
        purified_images_rescaled = (purified_images + 1) / 2
        return purified_images_rescaled.to(DEVICE)

# =============================================================================
# 5. Trainer Implementations
# =============================================================================

class BaseTrainer:
    def __init__(self, model, train_loader, test_loader, num_classes, defense_name, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.defense_name = defense_name
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.history = defaultdict(list)
        self.attack_params = {}

    def train(self, epochs, lr, attack_params): raise NotImplementedError

    def evaluate(self, attack=None):
        self.model.eval()
        correct_clean, total_clean = 0, 0
        correct_poison, total_poison = 0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
                
                if attack:
                    non_target_mask = (labels != attack.target_class)
                    if torch.any(non_target_mask):
                        poisoned_images = attack.apply_trigger(images[non_target_mask].clone())
                        poisoned_labels = torch.full_like(labels[non_target_mask], attack.target_class)
                        outputs_poison = self.model(poisoned_images)
                        _, predicted_poison = torch.max(outputs_poison.data, 1)
                        total_poison += poisoned_labels.size(0)
                        correct_poison += (predicted_poison == poisoned_labels).sum().item()

        ca = 100 * correct_clean / total_clean if total_clean > 0 else 0
        asr = 100 * correct_poison / total_poison if total_poison > 0 else 0
        return ca, asr
    
    def get_log_prefix(self): return f"[{self.defense_name} | {self.attack_params.get('name', 'N/A')}]"

class VanillaTrainer(BaseTrainer):
    def train(self, epochs, lr, attack_params):
        self.attack_params = attack_params
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        print(f"\n{self.get_log_prefix()} Starting Vanilla Training...")

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, labels, _) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(self.train_loader)
            ca, asr = self.evaluate(attack_params.get('attack_instance'))
            self.history['loss'].append(epoch_loss)
            self.history['ca'].append(ca)
            self.history['asr'].append(asr)
            print(f"{self.get_log_prefix()} Epoch {epoch+1}/{epochs} -> Loss: {epoch_loss:.4f}, CA: {ca:.2f}%, ASR: {asr:.2f}%")

class UltraCPlusTrainer(BaseTrainer):
    def train(self, epochs, lr, attack_params):
        if not FUNCTORCH_AVAILABLE:
            print(f"Skipping {self.defense_name}: functorch not available.")
            return
        self.attack_params = attack_params
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        
        num_train_samples = len(self.train_loader.dataset)
        self.tracker = TrajectoryTracker(num_samples=num_train_samples)
        self.scoring_head = ScoringHead().to(self.device)
        self.cgs_manager = CGSManager(num_classes=self.num_classes, model=self.model)
        
        scoring_optimizer = optim.Adam(self.scoring_head.parameters(), lr=1e-3)
        self.per_sample_criterion = nn.CrossEntropyLoss(reduction='none')
        print(f"\n{self.get_log_prefix()} Starting ULTRA-C+ Training...")

        fmodel, params = torch.func.make_functional(self.model)
        def compute_loss_stateless(params, buffer, sample, target):
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)
            output = fmodel(params, batch)
            return self.criterion(output, targets)
        
        ft_compute_grad = grad(compute_loss_stateless)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

        for epoch in range(epochs):
            self.model.train()
            self.scoring_head.train()
            running_loss = 0.0
            for batch_idx, (inputs, labels, is_poison) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                sample_indices = self.train_loader.batch_sampler.sampler.indices[batch_idx * self.train_loader.batch_size : (batch_idx + 1) * self.train_loader.batch_size]

                outputs = self.model(inputs)
                per_sample_losses = self.per_sample_criterion(outputs, labels)
                main_loss = per_sample_losses.mean()

                features = self.tracker.update_and_get_features(per_sample_losses, sample_indices)
                anomaly_scores = self.scoring_head(features)

                is_poison = is_poison.to(self.device).float()
                score_loss = F.binary_cross_entropy(anomaly_scores, is_poison)
                scoring_optimizer.zero_grad()
                score_loss.backward(retain_graph=True)
                scoring_optimizer.step()

                per_sample_grads = ft_compute_sample_grad(list(self.model.parameters()), [], inputs, labels)
                
                clean_mask = (anomaly_scores < torch.quantile(anomaly_scores.detach(), 0.5))
                self.cgs_manager.update_ema(per_sample_grads, labels, clean_mask)
                
                score_thresh_steer = torch.quantile(anomaly_scores.detach(), 0.75)
                corrected_grads = self.cgs_manager.steer_gradients(per_sample_grads, labels, anomaly_scores, score_thresh_steer)

                optimizer.zero_grad()
                final_batch_grads = [g.mean(dim=0) for g in corrected_grads]
                for param, grad_val in zip(self.model.parameters(), final_batch_grads):
                    param.grad = grad_val
                optimizer.step()
                running_loss += main_loss.item()

            epoch_loss = running_loss / len(self.train_loader)
            ca, asr = self.evaluate(attack_params.get('attack_instance'))
            self.history['loss'].append(epoch_loss)
            self.history['ca'].append(ca)
            self.history['asr'].append(asr)
            print(f"{self.get_log_prefix()} Epoch {epoch+1}/{epochs} -> Loss: {epoch_loss:.4f}, CA: {ca:.2f}%, ASR: {asr:.2f}%")

class EpicTrainer(BaseTrainer):
    def train(self, epochs, lr, attack_params):
        if not FUNCTORCH_AVAILABLE:
            print(f"Skipping {self.defense_name}: functorch not available.")
            return
        self.attack_params = attack_params
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        print(f"\n{self.get_log_prefix()} Starting EPIC Training...")

        fmodel, params = torch.func.make_functional(self.model)
        def compute_loss_stateless(params, sample, target):
            output = fmodel(params, sample.unsqueeze(0))
            return self.criterion(output, target.unsqueeze(0))
        ft_compute_grad = grad(compute_loss_stateless)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, labels, _) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                if i % 10 == 0:
                    per_sample_grads = ft_compute_sample_grad(list(self.model.parameters()), inputs, labels)
                    flat_grads = torch.cat([g.view(g.shape[0], -1) for g in per_sample_grads], dim=1)
                    
                    try:
                        U, S, V = torch.svd(flat_grads)
                        k = 1
                        projected_flat_grads = torch.matmul(flat_grads, V[:, :k]).matmul(V[:, :k].T)
                        pointer = 0
                        for p in self.model.parameters():
                            num_params = p.numel()
                            p.grad = projected_flat_grads.mean(dim=0)[pointer:pointer+num_params].view_as(p)
                            pointer += num_params
                    except torch.linalg.LinAlgError:
                        loss.backward()
                else:
                    loss.backward()
                
                optimizer.step()
            
            epoch_loss = running_loss / len(self.train_loader)
            ca, asr = self.evaluate(attack_params.get('attack_instance'))
            self.history['loss'].append(epoch_loss)
            self.history['ca'].append(ca)
            self.history['asr'].append(asr)
            print(f"{self.get_log_prefix()} Epoch {epoch+1}/{epochs} -> Loss: {epoch_loss:.4f}, CA: {ca:.2f}%, ASR: {asr:.2f}%")

class FinePuningTrainer(BaseTrainer):
    def train(self, epochs, lr, attack_params):
        self.attack_params = attack_params
        vanilla_trainer = VanillaTrainer(self.model, self.train_loader, self.test_loader, self.num_classes, "FP_Stage1", self.device)
        vanilla_trainer.train(epochs, lr, attack_params)

        print(f"\n{self.get_log_prefix()} Starting Fine-tuning Stage...")
        
        # This logic for getting clean samples assumes PoisonedDataset is used
        clean_indices = [i for i, p_mask in enumerate(self.train_loader.dataset.poison_mask) if not p_mask]
        clean_subset = torch.utils.data.Subset(self.train_loader.dataset, clean_indices[:int(0.05 * len(clean_indices))])
        finetune_loader = DataLoader(clean_subset, batch_size=self.train_loader.batch_size, shuffle=True)

        optimizer = optim.SGD(self.model.parameters(), lr=lr/10, momentum=0.9)
        finetune_epochs = 3
        for epoch in range(finetune_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels, _ in finetune_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(finetune_loader) if len(finetune_loader) > 0 else 0
            ca, asr = self.evaluate(attack_params.get('attack_instance'))
            self.history['loss'].append(epoch_loss)
            self.history['ca'].append(ca)
            self.history['asr'].append(asr)
            print(f"{self.get_log_prefix()} Finetune Epoch {epoch+1}/{finetune_epochs} -> Loss: {epoch_loss:.4f}, CA: {ca:.2f}%, ASR: {asr:.2f}%")
