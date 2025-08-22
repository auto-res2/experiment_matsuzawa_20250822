"""
Training module for Dynamic Behavioral Cartography (DBC).
Implements the core DBC algorithm with dynamic graph repertoire and VAE-based emitters.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import diptest
from collections import defaultdict
import copy


class VAE(nn.Module):
    """Variational Autoencoder for behavior descriptor modeling."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def reconstruction_loss(self, recon_x, x):
        return nn.MSELoss()(recon_x, x)
    
    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class PolicyNetwork(nn.Module):
    """Multi-Layer Perceptron policy for robot control."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)
    
    def get_parameters(self):
        """Get flattened parameters for evolutionary optimization."""
        params = []
        for param in self.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)
    
    def set_parameters(self, params):
        """Set parameters from flattened vector."""
        param_idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param.data = params[param_idx:param_idx + param_size].reshape(param.shape)
            param_idx += param_size


class Niche:
    """Individual niche in the dynamic graph repertoire."""
    
    def __init__(self, niche_id: int, behavior_dim: int = 240, 
                 latent_dim: int = 16, max_elites: int = 10):
        self.niche_id = niche_id
        self.behavior_dim = behavior_dim
        self.latent_dim = latent_dim
        self.max_elites = max_elites
        
        self.elites = []  # List of (genome, fitness, behavior_descriptor) tuples
        self.behavior_descriptors = []
        
        self.local_vae = VAE(behavior_dim, latent_dim)
        self.vae_optimizer = optim.Adam(self.local_vae.parameters(), lr=1e-3)
        
        self.reconstruction_losses = []
        self.generation_created = 0
        self.last_split_generation = 0
    
    def add_elite(self, genome: np.ndarray, fitness: float, 
                  behavior_descriptor: np.ndarray):
        """Add a new elite solution to the niche."""
        elite = (genome, fitness, behavior_descriptor)
        
        if len(self.elites) < self.max_elites:
            self.elites.append(elite)
            self.behavior_descriptors.append(behavior_descriptor)
        else:
            worst_idx = min(range(len(self.elites)), 
                          key=lambda i: self.elites[i][1])
            if fitness > self.elites[worst_idx][1]:
                self.elites[worst_idx] = elite
                self.behavior_descriptors[worst_idx] = behavior_descriptor
    
    def train_local_vae(self, epochs: int = 10):
        """Train the local VAE on current elite behavior descriptors."""
        if len(self.behavior_descriptors) < 2:
            return
        
        bd_tensor = torch.FloatTensor(np.array(self.behavior_descriptors))
        
        for epoch in range(epochs):
            self.vae_optimizer.zero_grad()
            recon_bd, mu, logvar = self.local_vae(bd_tensor)
            
            recon_loss = self.local_vae.reconstruction_loss(recon_bd, bd_tensor)
            kl_loss = self.local_vae.kl_divergence(mu, logvar)
            total_loss = recon_loss + 0.1 * kl_loss
            
            total_loss.backward()
            self.vae_optimizer.step()
        
        self.reconstruction_losses.append(recon_loss.item())
    
    def should_split(self, min_elites: int = 5, significance_level: float = 0.05) -> bool:
        """Determine if this niche should split based on statistical test."""
        if len(self.elites) < min_elites:
            return False
        
        if len(self.reconstruction_losses) < 10:
            return False
        
        recent_losses = self.reconstruction_losses[-10:]
        loss_variance = np.var(recent_losses)
        if loss_variance > 0.01:  # Still improving
            return False
        
        bd_tensor = torch.FloatTensor(np.array(self.behavior_descriptors))
        with torch.no_grad():
            mu, _ = self.local_vae.encode(bd_tensor)
        
        pca = PCA(n_components=1)
        principal_component = pca.fit_transform(mu.numpy()).flatten()
        
        dip_stat, p_value = diptest.diptest(principal_component)
        
        return p_value < significance_level
    
    def split(self) -> Tuple['Niche', 'Niche']:
        """Split this niche into two child niches."""
        bd_tensor = torch.FloatTensor(np.array(self.behavior_descriptors))
        with torch.no_grad():
            mu, _ = self.local_vae.encode(bd_tensor)
        
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(mu.numpy())
        
        niche1 = Niche(self.niche_id * 2, self.behavior_dim, 
                      self.latent_dim, self.max_elites)
        niche2 = Niche(self.niche_id * 2 + 1, self.behavior_dim, 
                      self.latent_dim, self.max_elites)
        
        for i, (elite, bd) in enumerate(zip(self.elites, self.behavior_descriptors)):
            if cluster_labels[i] == 0:
                niche1.add_elite(elite[0], elite[1], elite[2])
            else:
                niche2.add_elite(elite[0], elite[1], elite[2])
        
        return niche1, niche2


class DynamicGraphRepertoire:
    """Dynamic graph repertoire for DBC algorithm."""
    
    def __init__(self, behavior_dim: int = 240, global_latent_dim: int = 32):
        self.behavior_dim = behavior_dim
        self.global_latent_dim = global_latent_dim
        
        self.graph = nx.Graph()
        self.niches = {}  # niche_id -> Niche object
        self.next_niche_id = 1
        
        self.global_vae = VAE(behavior_dim, global_latent_dim, hidden_dim=256)
        self.global_optimizer = optim.Adam(self.global_vae.parameters(), lr=1e-3)
        
        self.global_elite_buffer = []
        self.max_buffer_size = 1000
        
        self.generation = 0
        self.split_history = []
        self.merge_history = []
    
    def add_niche(self, niche: Niche):
        """Add a new niche to the repertoire."""
        self.niches[niche.niche_id] = niche
        self.graph.add_node(niche.niche_id)
        niche.generation_created = self.generation
    
    def remove_niche(self, niche_id: int):
        """Remove a niche from the repertoire."""
        if niche_id in self.niches:
            del self.niches[niche_id]
            self.graph.remove_node(niche_id)
    
    def update_global_buffer(self):
        """Update global elite buffer with samples from all niches."""
        new_samples = []
        for niche in self.niches.values():
            for _, _, bd in niche.elites:
                new_samples.append(bd)
        
        self.global_elite_buffer.extend(new_samples)
        
        if len(self.global_elite_buffer) > self.max_buffer_size:
            self.global_elite_buffer = self.global_elite_buffer[-self.max_buffer_size:]
    
    def train_global_vae(self, epochs: int = 20):
        """Train the Global Coordinator VAE."""
        if len(self.global_elite_buffer) < 10:
            return
        
        bd_tensor = torch.FloatTensor(np.array(self.global_elite_buffer))
        
        for epoch in range(epochs):
            self.global_optimizer.zero_grad()
            recon_bd, mu, logvar = self.global_vae(bd_tensor)
            
            recon_loss = self.global_vae.reconstruction_loss(recon_bd, bd_tensor)
            kl_loss = self.global_vae.kl_divergence(mu, logvar)
            total_loss = recon_loss + 0.1 * kl_loss
            
            total_loss.backward()
            self.global_optimizer.step()
    
    def update_graph_edges(self):
        """Update graph edges based on Wasserstein distances in global latent space."""
        niche_ids = list(self.niches.keys())
        
        self.graph.clear_edges()
        
        for i, niche_id1 in enumerate(niche_ids):
            for niche_id2 in niche_ids[i+1:]:
                distance = self.compute_niche_distance(niche_id1, niche_id2)
                
                if distance < 2.0:  # Threshold for connectivity
                    self.graph.add_edge(niche_id1, niche_id2, weight=distance)
    
    def compute_niche_distance(self, niche_id1: int, niche_id2: int) -> float:
        """Compute Wasserstein distance between two niches in global latent space."""
        niche1 = self.niches[niche_id1]
        niche2 = self.niches[niche_id2]
        
        if len(niche1.behavior_descriptors) == 0 or len(niche2.behavior_descriptors) == 0:
            return float('inf')
        
        bd1_tensor = torch.FloatTensor(np.array(niche1.behavior_descriptors))
        bd2_tensor = torch.FloatTensor(np.array(niche2.behavior_descriptors))
        
        with torch.no_grad():
            mu1, _ = self.global_vae.encode(bd1_tensor)
            mu2, _ = self.global_vae.encode(bd2_tensor)
        
        pca1 = PCA(n_components=1)
        pca2 = PCA(n_components=1)
        
        pc1 = pca1.fit_transform(mu1.numpy()).flatten()
        pc2 = pca2.fit_transform(mu2.numpy()).flatten()
        
        return wasserstein_distance(pc1, pc2)
    
    def check_for_merges(self, merge_threshold: float = 0.5) -> List[Tuple[int, int]]:
        """Check for niches that should be merged."""
        merge_candidates = []
        
        for edge in self.graph.edges(data=True):
            niche_id1, niche_id2, data = edge
            distance = data['weight']
            
            if distance < merge_threshold:
                merge_candidates.append((niche_id1, niche_id2))
        
        return merge_candidates
    
    def merge_niches(self, niche_id1: int, niche_id2: int) -> Niche:
        """Merge two niches into a single niche."""
        niche1 = self.niches[niche_id1]
        niche2 = self.niches[niche_id2]
        
        merged_niche = Niche(self.next_niche_id, self.behavior_dim, 
                           max(niche1.latent_dim, niche2.latent_dim))
        self.next_niche_id += 1
        
        all_elites = niche1.elites + niche2.elites
        
        all_elites.sort(key=lambda x: x[1], reverse=True)
        for elite in all_elites[:merged_niche.max_elites]:
            merged_niche.add_elite(elite[0], elite[1], elite[2])
        
        self.remove_niche(niche_id1)
        self.remove_niche(niche_id2)
        self.add_niche(merged_niche)
        
        self.merge_history.append((niche_id1, niche_id2, merged_niche.niche_id, self.generation))
        
        return merged_niche
    
    def step(self):
        """Perform one step of the DBC algorithm."""
        self.generation += 1
        
        for niche in self.niches.values():
            niche.train_local_vae()
        
        self.update_global_buffer()
        self.train_global_vae()
        
        if self.generation % 10 == 0:
            self.update_graph_edges()
        
        niches_to_split = []
        for niche in self.niches.values():
            if niche.should_split():
                niches_to_split.append(niche.niche_id)
        
        for niche_id in niches_to_split:
            if niche_id in self.niches:  # Check if still exists
                niche = self.niches[niche_id]
                niche1, niche2 = niche.split()
                
                self.remove_niche(niche_id)
                self.add_niche(niche1)
                self.add_niche(niche2)
                
                self.split_history.append((niche_id, niche1.niche_id, niche2.niche_id, self.generation))
        
        if self.generation % 20 == 0:
            merge_candidates = self.check_for_merges()
            for niche_id1, niche_id2 in merge_candidates:
                if niche_id1 in self.niches and niche_id2 in self.niches:
                    self.merge_niches(niche_id1, niche_id2)


class TopologyAwareEmitter:
    """Emitter that leverages graph topology for intelligent exploration."""
    
    def __init__(self, repertoire: DynamicGraphRepertoire, genome_size: int = 256):
        self.repertoire = repertoire
        self.genome_size = genome_size
        self.mutation_strength = 0.1
        
        self.frontier_archive = []
        self.max_frontier_size = 100
    
    def emit(self, batch_size: int = 50) -> List[np.ndarray]:
        """Generate new candidate solutions using hybrid strategy."""
        candidates = []
        
        exploitation_ratio = 0.4
        bridging_ratio = 0.4
        frontier_ratio = 0.2
        
        n_exploitation = int(batch_size * exploitation_ratio)
        n_bridging = int(batch_size * bridging_ratio)
        n_frontier = batch_size - n_exploitation - n_bridging
        
        candidates.extend(self.exploitation_emit(n_exploitation))
        
        candidates.extend(self.bridging_emit(n_bridging))
        
        candidates.extend(self.frontier_emit(n_frontier))
        
        return candidates
    
    def exploitation_emit(self, n_candidates: int) -> List[np.ndarray]:
        """Generate candidates by mutating elites within niches."""
        candidates = []
        
        if not self.repertoire.niches:
            return [np.random.randn(self.genome_size) * 0.1 for _ in range(n_candidates)]
        
        for _ in range(n_candidates):
            niche = np.random.choice(list(self.repertoire.niches.values()))
            
            if niche.elites:
                elite_idx = np.random.randint(len(niche.elites))
                elite = niche.elites[elite_idx]
                genome = elite[0]
                
                mutation = np.random.randn(len(genome)) * self.mutation_strength
                candidate = genome + mutation
                candidates.append(candidate)
            else:
                candidates.append(np.random.randn(self.genome_size) * 0.1)
        
        return candidates
    
    def bridging_emit(self, n_candidates: int) -> List[np.ndarray]:
        """Generate candidates by interpolating between connected niches."""
        candidates = []
        
        edges = list(self.repertoire.graph.edges())
        if not edges:
            return self.exploitation_emit(n_candidates)
        
        for _ in range(n_candidates):
            edge_idx = np.random.choice(len(edges))
            edge = edges[edge_idx]
            niche_id1, niche_id2 = edge[0], edge[1]
            
            niche1 = self.repertoire.niches[niche_id1]
            niche2 = self.repertoire.niches[niche_id2]
            
            if niche1.elites and niche2.elites:
                elite1_idx = np.random.randint(len(niche1.elites))
                elite2_idx = np.random.randint(len(niche2.elites))
                elite1 = niche1.elites[elite1_idx]
                elite2 = niche2.elites[elite2_idx]
                
                bd1 = torch.FloatTensor(elite1[2]).unsqueeze(0)
                bd2 = torch.FloatTensor(elite2[2]).unsqueeze(0)
                
                with torch.no_grad():
                    mu1, _ = self.repertoire.global_vae.encode(bd1)
                    mu2, _ = self.repertoire.global_vae.encode(bd2)
                    
                    alpha = np.random.random()
                    interpolated_latent = alpha * mu1 + (1 - alpha) * mu2
                    
                    decoded_bd = self.repertoire.global_vae.decode(interpolated_latent)
                
                genome1, genome2 = elite1[0], elite2[0]
                candidate = alpha * genome1 + (1 - alpha) * genome2
                
                mutation = np.random.randn(len(candidate)) * self.mutation_strength * 0.5
                candidate += mutation
                
                candidates.append(candidate)
            else:
                candidates.append(np.random.randn(self.genome_size) * 0.1)
        
        return candidates
    
    def frontier_emit(self, n_candidates: int) -> List[np.ndarray]:
        """Generate candidates from frontier archive."""
        candidates = []
        
        if not self.frontier_archive:
            return [np.random.randn(self.genome_size) * 0.1 for _ in range(n_candidates)]
        
        for _ in range(n_candidates):
            frontier_idx = np.random.randint(len(self.frontier_archive))
            frontier_genome = self.frontier_archive[frontier_idx]
            mutation = np.random.randn(len(frontier_genome)) * self.mutation_strength
            candidate = frontier_genome + mutation
            candidates.append(candidate)
        
        return candidates
    
    def update_frontier_archive(self, genome: np.ndarray, behavior_descriptor: np.ndarray):
        """Update frontier archive with solutions poorly reconstructed by local VAEs."""
        bd_tensor = torch.FloatTensor(behavior_descriptor).unsqueeze(0)
        
        local_reconstruction_errors = []
        for niche in self.repertoire.niches.values():
            with torch.no_grad():
                recon_bd, _, _ = niche.local_vae(bd_tensor)
                error = torch.mean((recon_bd - bd_tensor) ** 2).item()
                local_reconstruction_errors.append(error)
        
        with torch.no_grad():
            global_recon_bd, _, _ = self.repertoire.global_vae(bd_tensor)
            global_error = torch.mean((global_recon_bd - bd_tensor) ** 2).item()
        
        if local_reconstruction_errors and global_error < np.mean(local_reconstruction_errors):
            self.frontier_archive.append(genome)
            
            if len(self.frontier_archive) > self.max_frontier_size:
                self.frontier_archive.pop(0)


def train_dbc(num_generations: int = 100, population_size: int = 100) -> DynamicGraphRepertoire:
    """Main training loop for DBC algorithm."""
    
    repertoire = DynamicGraphRepertoire()
    emitter = TopologyAwareEmitter(repertoire)
    
    initial_niche = Niche(repertoire.next_niche_id, behavior_dim=240)
    repertoire.next_niche_id += 1
    repertoire.add_niche(initial_niche)
    
    print(f"Starting DBC training for {num_generations} generations...")
    
    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations}")
        
        candidates = emitter.emit(population_size)
        
        for candidate in candidates:
            fitness = np.random.random()  # Placeholder fitness
            behavior_descriptor = np.random.randn(240)  # Placeholder BD
            
            if repertoire.niches:
                niche = list(repertoire.niches.values())[0]  # Simplified assignment
                niche.add_elite(candidate, fitness, behavior_descriptor)
                
                emitter.update_frontier_archive(candidate, behavior_descriptor)
        
        repertoire.step()
        
        if (generation + 1) % 10 == 0:
            print(f"  Niches: {len(repertoire.niches)}")
            print(f"  Splits: {len(repertoire.split_history)}")
            print(f"  Merges: {len(repertoire.merge_history)}")
    
    return repertoire


if __name__ == "__main__":
    print("Testing DBC training components...")
    
    vae = VAE(input_dim=240, latent_dim=16)
    test_input = torch.randn(10, 240)
    recon, mu, logvar = vae(test_input)
    print(f"VAE test - Input: {test_input.shape}, Reconstruction: {recon.shape}")
    
    niche = Niche(1, behavior_dim=240)
    test_genome = np.random.randn(256)
    test_bd = np.random.randn(240)
    niche.add_elite(test_genome, 0.5, test_bd)
    print(f"Niche test - Elites: {len(niche.elites)}")
    
    repertoire = train_dbc(num_generations=5, population_size=10)
    print(f"Training test completed - Final niches: {len(repertoire.niches)}")
