import torch
from torch_geometric.loader import GraphSAINTNodeSampler
from torch_geometric.utils import add_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
import os

def get_dataset_and_sampler(name, device, batch_size=2048, num_steps=5):
    """Load dataset and create GraphSAINT sampler."""
    print(f"Loading dataset: {name}")
    
    os.makedirs('./data', exist_ok=True)
    
    dataset = PygNodePropPredDataset(name=name, root='./data')
    data = dataset[0]
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    data = data.to(device)

    loader = GraphSAINTNodeSampler(
        data, 
        batch_size=batch_size, 
        num_steps=num_steps, 
        sample_coverage=0, 
        save_dir=dataset.processed_dir
    )
    
    split_idx = dataset.get_idx_split()
    return data, loader, split_idx

def prepare_datasets(device):
    """Prepare both datasets used in the experiment."""
    datasets = {}
    
    try:
        data_arxiv, loader_arxiv, split_arxiv = get_dataset_and_sampler("ogbn-arxiv", device)
        datasets["ogbn-arxiv"] = {
            'data': data_arxiv,
            'loader': loader_arxiv,
            'split_idx': split_arxiv
        }
        print(f"ogbn-arxiv loaded: {data_arxiv.num_nodes} nodes, {data_arxiv.num_edges} edges")
    except Exception as e:
        print(f"Failed to load ogbn-arxiv: {e}")
        datasets["ogbn-arxiv"] = None
    
    try:
        data_products, loader_products, split_products = get_dataset_and_sampler("ogbn-products", device)
        datasets["ogbn-products"] = {
            'data': data_products,
            'loader': loader_products,
            'split_idx': split_products
        }
        print(f"ogbn-products loaded: {data_products.num_nodes} nodes, {data_products.num_edges} edges")
    except Exception as e:
        print(f"Failed to load ogbn-products (this is expected on limited hardware): {e}")
        datasets["ogbn-products"] = None
    
    return datasets
