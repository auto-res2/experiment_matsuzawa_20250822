import time
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
from tqdm import tqdm
import warnings
import math
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """Encodes high-dimensional data X to latent space Z."""
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

    def forward(self, x):
        params = self.net(x)
        mu, log_var = torch.chunk(params, 2, dim=1)
        return Normal(mu, torch.exp(0.5 * log_var))

class StructuredDecoder(nn.Module):
    """Decodes latent Z to X, respecting the causal graph structure."""
    def __init__(self, graph, latent_dim, hidden_dim=64):
        super().__init__()
        self.graph = graph
        self.latent_dim = latent_dim
        self.node_mlps = nn.ModuleDict()

        for node in graph.nodes():
            num_parents = len(list(graph.predecessors(node)))
            input_dim = num_parents + latent_dim
            self.node_mlps[node.replace(".", "_")] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2) # mu and log_std
            )
    
    def forward_single_node(self, node, parent_values, z):
        # parent_values shape: [batch_size, num_parents]
        # z shape: [batch_size, latent_dim]
        if parent_values.nelement() == 0:
             inputs = z
        else:
             inputs = torch.cat([parent_values, z], dim=1)

        params = self.node_mlps[node.replace(".", "_")](inputs)
        mu, log_std = torch.chunk(params, 2, dim=1)
        return Normal(mu, torch.exp(log_std))

class CausalFactorSampler:
    """A DoSampler based on a Structured Causal Variational Autoencoder."""
    def __init__(self, data, graph, num_latent_dims, epochs=100, batch_size=128, lr=1e-3):
        print("Initializing and training CausalFactorSampler...")
        self.graph = graph
        self.latent_dim = num_latent_dims
        self.topological_order = list(nx.topological_sort(graph))
        self.data_dim = data.shape[1]
        self.device = DEVICE

        # Initialize models
        self.encoder = Encoder(self.data_dim, self.latent_dim).to(self.device)
        self.decoder = StructuredDecoder(self.graph, self.latent_dim).to(self.device)
        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(self.params, lr=lr)

        # Prepare data
        self.data_tensor = torch.tensor(data.values, dtype=torch.float32).to(self.device)
        self.train_loader = torch.utils.data.DataLoader(self.data_tensor, batch_size=batch_size, shuffle=True)

        # Train the SC-VAE
        self.train(epochs)

    def train(self, epochs):
        self.encoder.train()
        self.decoder.train()
        pbar = tqdm(range(epochs), desc="Training SC-VAE")
        for epoch in pbar:
            total_loss = 0
            for x_batch in self.train_loader:
                self.optimizer.zero_grad()
                
                q_z_x = self.encoder(x_batch)
                z = q_z_x.rsample()
                p_z = Normal(torch.zeros_like(z), torch.ones_like(z))
                
                kl_div = kl_divergence(q_z_x, p_z).sum(dim=1).mean()

                recon_log_prob = 0
                parent_map = {node: list(self.graph.predecessors(node)) for node in self.topological_order}
                node_data_map = {node: x_batch[:, i] for i, node in enumerate(self.graph.nodes())}

                for node in self.topological_order:
                    parents = parent_map[node]
                    if parents:
                        parent_values = torch.stack([node_data_map[p] for p in parents], dim=1)
                    else:
                        parent_values = torch.empty(x_batch.size(0), 0, device=self.device)
                    
                    p_x_node = self.decoder.forward_single_node(node, parent_values, z)
                    recon_log_prob += p_x_node.log_prob(node_data_map[node].unsqueeze(1))
                
                recon_loss = -recon_log_prob.sum(dim=1).mean()
                
                loss = recon_loss + kl_div
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            pbar.set_postfix({"ELBO Loss": f"{avg_loss:.4f}"})

    @torch.no_grad()
    def sample(self, num_samples, intervention_var, intervention_val):
        self.encoder.eval()
        self.decoder.eval()

        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        generated_samples = {}

        generated_samples[intervention_var] = torch.full(
            (num_samples, 1), intervention_val, device=self.device
        )

        for node in self.topological_order:
            if node in generated_samples:
                continue

            parents = list(self.graph.predecessors(node))
            
            if parents:
                parent_values = torch.cat([generated_samples[p] for p in parents], dim=1)
            else:
                parent_values = torch.empty(num_samples, 0, device=self.device)

            node_dist = self.decoder.forward_single_node(node, parent_values, z)
            generated_samples[node] = node_dist.sample()

        df = pd.DataFrame(
            {k: v.cpu().numpy().flatten() for k, v in generated_samples.items()}
        )
        return df[list(self.graph.nodes())]

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
        }, path)
        print(f"Model saved to {path}")

class McmcSamplerPlaceholder:
    """Simulates a slow MCMC sampler to demonstrate scaling issues."""
    def __init__(self, data, graph):
        print("Initializing McmcSamplerPlaceholder. NOTE: This is a placeholder and simulates time.")
        self.d = data.shape[1]
        self.time_factor = 0.000005 

    def sample(self, num_samples, intervention_var, intervention_val):
        sleep_duration = self.time_factor * self.d * num_samples
        time.sleep(sleep_duration)
        return pd.DataFrame(np.random.randn(num_samples, self.d), columns=[f'X{i}' for i in range(self.d)])

class KernelDensitySamplerPlaceholder:
    """Simulates a Kernel Density sampler using simple rejection sampling."""
    def __init__(self, data, graph):
        print("Initializing KernelDensitySamplerPlaceholder. This will be slow for d > 10.")
        self.data = data
        self.graph = graph
        self.epsilon = 0.1 
        self.max_tries = 100000 

    def sample(self, num_samples, intervention_var, intervention_val):
        if self.data.shape[1] > 50:
             raise TimeoutError("KDE Sampler is infeasible for d > 50.")
             
        samples = []
        intervention_col_idx = self.data.columns.get_loc(intervention_var)
        data_numpy = self.data.to_numpy()

        for _ in range(num_samples):
            tries = 0
            while tries < self.max_tries:
                candidate = data_numpy[np.random.randint(0, len(data_numpy))]
                if abs(candidate[intervention_col_idx] - intervention_val) < self.epsilon * self.data[intervention_var].std():
                    final_sample = candidate.copy()
                    final_sample[intervention_col_idx] = intervention_val
                    samples.append(final_sample)
                    break
                tries += 1
            if tries == self.max_tries:
                warnings.warn(f"KDE sampler max tries reached for one sample. Results may be biased.")

        return pd.DataFrame(samples, columns=self.data.columns)