import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import math
import random
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_scm_data(n_samples, d, data_dir, seed, n_latent=None, p=2.0):
    """
    Generates synthetic data from a non-linear SCM with latent confounders and saves it.
    """
    if n_latent is None:
        n_latent = math.ceil(d / 10)

    while True:
        g = nx.erdos_renyi_graph(d, p / d, seed=seed, directed=True)
        if nx.is_directed_acyclic_graph(g):
            break
    
    graph = nx.DiGraph()
    graph.add_nodes_from([f'X{i}' for i in range(d)])
    graph.add_edges_from([(f'X{u}', f'X{v}') for u, v in g.edges()])
    topological_order = list(nx.topological_sort(graph))

    # Define functional relationships (MLPs)
    scm_functions = {}
    for node in topological_order:
        parents = list(graph.predecessors(node))
        num_latent_parents = np.random.binomial(n_latent, 1.0/5.0)
        latent_parents_indices = np.random.choice(n_latent, num_latent_parents, replace=False)
        
        input_dim = len(parents) + len(latent_parents_indices)
        if input_dim > 0:
            func = nn.Sequential(
                nn.Linear(input_dim, 20),
                nn.SiLU(),
                nn.Linear(20, 20),
                nn.SiLU(),
                nn.Linear(20, 1)
            ).to(DEVICE)
            scm_functions[node] = (func, parents, [f'Z{i}' for i in latent_parents_indices])

    data = {}
    latents = {f'Z{i}': torch.randn(n_samples, 1).to(DEVICE) for i in range(n_latent)}
    
    with torch.no_grad():
        for node in topological_order:
            if node in scm_functions:
                func, parents, latent_parents = scm_functions[node]
                parent_data = [data[p] for p in parents]
                latent_data = [latents[z] for z in latent_parents]
                
                if parent_data or latent_data:
                    inputs = torch.cat(parent_data + latent_data, dim=1)
                    mean = func(inputs)
                else:
                    mean = torch.zeros(n_samples, 1, device=DEVICE)
            else:
                mean = torch.zeros(n_samples, 1, device=DEVICE)

            noise = torch.randn(n_samples, 1).to(DEVICE)
            data[node] = mean + noise

    df_data = pd.DataFrame({k: v.cpu().numpy().flatten() for k, v in data.items()})
    df_data = df_data[topological_order]

    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, f"synthetic_data_d{d}.csv")
    graph_path = os.path.join(data_dir, f"graph_d{d}.gml")
    
    df_data.to_csv(data_path, index=False)
    nx.write_gml(graph, graph_path)
    
    print(f"Generated and saved data for d={d} to {data_dir}")
    
    return graph, df_data