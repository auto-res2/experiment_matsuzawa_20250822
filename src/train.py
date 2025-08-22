import torch
import torch.nn as nn
import numpy as np
import time

class MockGatedBlock(nn.Module):
    """A mock block with a fixed computational cost."""
    def __init__(self, gflops, base_latency_ms, accuracy_gain):
        super().__init__()
        self.gflops = gflops
        self.base_latency_ms = base_latency_ms
        self.accuracy_gain = accuracy_gain
        # A dummy layer to make sure the model has parameters
        self.layer = nn.Linear(10, 10)

    def forward(self, x, difficulty):
        # Simulate computation
        time.sleep(self.base_latency_ms / 1000.0)  # Convert ms to s
        # Simulate accuracy gain, less gain for harder samples
        # A random factor is added to make accuracy non-deterministic
        gain = self.accuracy_gain * (1 - 0.5 * difficulty) + np.random.uniform(-0.01, 0.01)
        return x + gain

class MetaBAPLResNet(nn.Module):
    """Simulates the Meta-BAPL model that adapts to a budget vector."""
    def __init__(self, num_blocks=5):
        super().__init__()
        self.blocks = nn.ModuleList([
            MockGatedBlock(gflops=0.5, base_latency_ms=5, accuracy_gain=0.15) for _ in range(num_blocks)
        ])
        # Budget encoder: maps [latency, gflops] to an embedding
        self.budget_encoder = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, num_blocks))
        self.base_accuracy = 0.20 # Accuracy with zero blocks

    def forward(self, x, difficulty, budget):
        # budget is a 2D tensor: [target_latency_ms, target_gflops]
        # Normalize budget for encoder (simple scaling)
        norm_budget = budget / torch.tensor([50.0, 2.5], device=budget.device)
        
        # Policy generation from budget
        policy_logits = self.budget_encoder(norm_budget)
        
        total_gflops = 0
        gflops_budget = budget[1]
        
        accuracy_score = torch.tensor(self.base_accuracy, device=x.device)

        for i, block in enumerate(self.blocks):
            # The policy logits could modulate this decision.
            # We simplify: execute if we are within GFLOPs budget and policy is positive.
            if total_gflops + block.gflops <= gflops_budget and policy_logits[i] > 0:
                accuracy_score = block(accuracy_score, difficulty)
                total_gflops += block.gflops

        return accuracy_score, total_gflops

class StaticModel(nn.Module):
    """Simulates a static model from the Model Zoo."""
    def __init__(self, num_active_blocks):
        super().__init__()
        self.num_active_blocks = num_active_blocks
        self.blocks = nn.ModuleList([
            MockGatedBlock(gflops=0.5, base_latency_ms=5, accuracy_gain=0.15) for _ in range(num_active_blocks)
        ])
        self.base_accuracy = 0.20
        self.total_gflops = sum(b.gflops for b in self.blocks)

    def forward(self, x, difficulty):
        accuracy_score = torch.tensor(self.base_accuracy, device=x.device)
        for block in self.blocks:
            accuracy_score = block(accuracy_score, difficulty)
        return accuracy_score, self.total_gflops

class ConfidenceDynamicNet(nn.Module):
    """Simulates a confidence-based dynamic network."""
    def __init__(self, num_blocks=5):
        super().__init__()
        self.blocks = nn.ModuleList([
            MockGatedBlock(gflops=0.5, base_latency_ms=5, accuracy_gain=0.15) for _ in range(num_blocks)
        ])
        self.base_accuracy = 0.20

    def forward(self, x, difficulty, confidence_threshold):
        accuracy_score = torch.tensor(self.base_accuracy, device=x.device)
        total_gflops = 0
        
        # Simulate confidence score. Harder samples yield lower confidence.
        # More blocks increase confidence.
        for block in self.blocks:
            current_confidence = torch.sigmoid(accuracy_score * 5 - 2.5) * (1 - 0.4 * difficulty)
            if current_confidence > confidence_threshold:
                break # Early exit
            accuracy_score = block(accuracy_score, difficulty)
            total_gflops += block.gflops
            
        return accuracy_score, total_gflops

def get_models(config):
    """Instantiates all models based on the config."""
    cfg_models = config['models']
    
    metabapl_model = MetaBAPLResNet(num_blocks=cfg_models['metabapl']['num_blocks'])
    
    model_zoo = {
        item['name']: StaticModel(num_active_blocks=item['active_blocks'])
        for item in cfg_models['model_zoo']
    }
    
    policy_zoo = {
        item['name']: StaticModel(num_active_blocks=item['active_blocks'])
        for item in cfg_models['policy_zoo']
    }
    
    confidence_model = ConfidenceDynamicNet(num_blocks=cfg_models['confidence']['num_blocks'])
    
    print("Models initialized.")
    return metabapl_model, model_zoo, policy_zoo, confidence_model
