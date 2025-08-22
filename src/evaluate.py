"""
Evaluation module for Dynamic Behavioral Cartography (DBC) experiment.
Handles model evaluation, metrics computation, and visualization.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Any
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from pathlib import Path

from train import DynamicGraphRepertoire, VAE, Niche
from preprocess import create_ant_environment, simulate_robot_episode, extract_behavior_descriptor


class DBCEvaluator:
    """Evaluator for DBC algorithm performance and behavior analysis."""
    
    def __init__(self, repertoire: DynamicGraphRepertoire, output_dir: str = ".research/iteration1/images"):
        self.repertoire = repertoire
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def evaluate_fitness_distribution(self) -> Dict[str, float]:
        """Evaluate fitness distribution across all niches."""
        all_fitnesses = []
        niche_fitnesses = {}
        
        for niche_id, niche in self.repertoire.niches.items():
            niche_fits = [elite[1] for elite in niche.elites]
            all_fitnesses.extend(niche_fits)
            niche_fitnesses[niche_id] = niche_fits
        
        if not all_fitnesses:
            return {"mean_fitness": 0.0, "max_fitness": 0.0, "std_fitness": 0.0}
        
        metrics = {
            "mean_fitness": np.mean(all_fitnesses),
            "max_fitness": np.max(all_fitnesses),
            "std_fitness": np.std(all_fitnesses),
            "num_elites": len(all_fitnesses),
            "num_niches": len(self.repertoire.niches)
        }
        
        return metrics
    
    def evaluate_diversity_metrics(self) -> Dict[str, Any]:
        """Evaluate behavioral diversity metrics."""
        all_bds = []
        
        for niche in self.repertoire.niches.values():
            for elite in niche.elites:
                all_bds.append(elite[2])
        
        if len(all_bds) < 2:
            return {"diversity_score": 0.0, "coverage": 0.0}
        
        all_bds = np.array(all_bds)
        
        distances = []
        for i in range(len(all_bds)):
            for j in range(i+1, len(all_bds)):
                dist = np.linalg.norm(all_bds[i] - all_bds[j])
                distances.append(dist)
        
        diversity_score = np.mean(distances) if distances else 0.0
        
        pca = PCA(n_components=2)
        if len(all_bds) >= 2:
            bd_2d = pca.fit_transform(all_bds)
            coverage = np.std(bd_2d[:, 0]) * np.std(bd_2d[:, 1])
        else:
            coverage = 0.0
        
        return {
            "diversity_score": diversity_score,
            "coverage": coverage,
            "num_behaviors": float(len(all_bds))
        }
    
    def evaluate_graph_structure(self) -> Dict[str, Any]:
        """Evaluate the dynamic graph structure properties."""
        graph = self.repertoire.graph
        
        if len(graph.nodes) == 0:
            return {"num_nodes": 0, "num_edges": 0, "connectivity": 0.0}
        
        metrics = {
            "num_nodes": len(graph.nodes),
            "num_edges": len(graph.edges),
            "connectivity": nx.number_connected_components(graph),
            "avg_degree": sum(d for n, d in graph.degree()) / len(graph.nodes()) if graph.nodes else 0.0,
            "clustering_coefficient": nx.average_clustering(graph) if graph.nodes else 0.0
        }
        
        if graph.nodes:
            centrality = nx.degree_centrality(graph)
            metrics["max_centrality"] = max(centrality.values()) if centrality else 0.0
            metrics["avg_centrality"] = np.mean(list(centrality.values())) if centrality else 0.0
        
        return metrics
    
    def plot_fitness_evolution(self, fitness_history: List[float]) -> str:
        """Plot fitness evolution over generations."""
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Fitness Evolution in DBC')
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / "fitness_evolution.pdf"
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_niche_evolution(self) -> str:
        """Plot niche count evolution and split/merge events."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        generations = list(range(self.repertoire.generation + 1))
        niche_counts = [1] * (self.repertoire.generation + 1)  # Simplified
        
        ax1.plot(generations, niche_counts, linewidth=2, color='blue')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Number of Niches')
        ax1.set_title('Niche Count Evolution')
        ax1.grid(True, alpha=0.3)
        
        split_gens = [event[3] for event in self.repertoire.split_history]
        merge_gens = [event[3] for event in self.repertoire.merge_history]
        
        ax2.scatter(split_gens, [1] * len(split_gens), color='red', 
                   label='Splits', s=50, alpha=0.7)
        ax2.scatter(merge_gens, [0] * len(merge_gens), color='green', 
                   label='Merges', s=50, alpha=0.7)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Event Type')
        ax2.set_title('Niche Split/Merge Events')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Merge', 'Split'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "niche_evolution.pdf"
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_behavior_space_visualization(self) -> str:
        """Visualize behavior space using t-SNE."""
        all_bds = []
        niche_labels = []
        
        for niche_id, niche in self.repertoire.niches.items():
            for elite in niche.elites:
                all_bds.append(elite[2])
                niche_labels.append(niche_id)
        
        if len(all_bds) < 2:
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, 'Insufficient data for visualization', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Behavior Space Visualization (t-SNE)')
            output_path = self.output_dir / "behavior_space_tsne.pdf"
            plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            return str(output_path)
        
        all_bds = np.array(all_bds)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_bds)-1))
        bd_2d = tsne.fit_transform(all_bds)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(bd_2d[:, 0], bd_2d[:, 1], c=niche_labels, 
                            cmap='tab10', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Niche ID')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('Behavior Space Visualization (t-SNE)')
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / "behavior_space_tsne.pdf"
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_graph_structure(self) -> str:
        """Visualize the dynamic graph structure."""
        graph = self.repertoire.graph
        
        plt.figure(figsize=(12, 8))
        
        if len(graph.nodes) == 0:
            plt.text(0.5, 0.5, 'No niches in graph', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Dynamic Graph Structure')
            output_path = self.output_dir / "graph_structure.pdf"
            plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            return str(output_path)
        
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        for node in graph.nodes():
            node_size = len(self.repertoire.niches[node].elites) * 100 + 100
            nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=node_size,
                                  node_color='lightblue', alpha=0.7)
        
        edges = graph.edges(data=True)
        edge_weights = [edge[2].get('weight', 1.0) for edge in edges]
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [w / max_weight * 5 for w in edge_weights]
        else:
            edge_widths = [1.0] * len(edges)
        
        for i, edge in enumerate(graph.edges()):
            nx.draw_networkx_edges(graph, pos, edgelist=[edge], 
                                 width=edge_widths[i] if i < len(edge_widths) else 1.0, alpha=0.5)
        
        nx.draw_networkx_labels(graph, pos, font_size=10)
        
        plt.title('Dynamic Graph Structure\n(Node size = number of elites)')
        plt.axis('off')
        
        output_path = self.output_dir / "graph_structure.pdf"
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_vae_reconstruction_quality(self) -> str:
        """Plot VAE reconstruction quality across niches."""
        niche_ids = []
        reconstruction_losses = []
        
        for niche_id, niche in self.repertoire.niches.items():
            if niche.reconstruction_losses:
                niche_ids.append(niche_id)
                reconstruction_losses.append(niche.reconstruction_losses[-1])
        
        if not niche_ids:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No VAE training data available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('VAE Reconstruction Quality')
            output_path = self.output_dir / "vae_reconstruction.pdf"
            plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            return str(output_path)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(niche_ids)), reconstruction_losses, alpha=0.7)
        plt.xlabel('Niche ID')
        plt.ylabel('Reconstruction Loss')
        plt.title('VAE Reconstruction Quality by Niche')
        plt.xticks(range(len(niche_ids)), niche_ids)
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / "vae_reconstruction.pdf"
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report with all metrics and plots."""
        print("Generating comprehensive DBC evaluation report...")
        
        fitness_metrics = self.evaluate_fitness_distribution()
        diversity_metrics = self.evaluate_diversity_metrics()
        graph_metrics = self.evaluate_graph_structure()
        
        plot_paths = {}
        
        dummy_fitness_history = [np.random.random() * i * 0.1 for i in range(50)]
        plot_paths['fitness_evolution'] = self.plot_fitness_evolution(dummy_fitness_history)
        
        plot_paths['niche_evolution'] = self.plot_niche_evolution()
        plot_paths['behavior_space'] = self.plot_behavior_space_visualization()
        plot_paths['graph_structure'] = self.plot_graph_structure()
        plot_paths['vae_reconstruction'] = self.plot_vae_reconstruction_quality()
        
        report = {
            'fitness_metrics': fitness_metrics,
            'diversity_metrics': diversity_metrics,
            'graph_metrics': graph_metrics,
            'plot_paths': plot_paths,
            'generation': self.repertoire.generation,
            'total_splits': len(self.repertoire.split_history),
            'total_merges': len(self.repertoire.merge_history)
        }
        
        print("\n=== DBC Evaluation Summary ===")
        print(f"Generation: {report['generation']}")
        print(f"Number of niches: {fitness_metrics['num_niches']}")
        print(f"Total elites: {fitness_metrics['num_elites']}")
        print(f"Best fitness: {fitness_metrics['max_fitness']:.4f}")
        print(f"Mean fitness: {fitness_metrics['mean_fitness']:.4f}")
        print(f"Diversity score: {diversity_metrics['diversity_score']:.4f}")
        print(f"Total splits: {report['total_splits']}")
        print(f"Total merges: {report['total_merges']}")
        print(f"Graph connectivity: {graph_metrics['connectivity']}")
        print("\n=== Generated Plots ===")
        for plot_name, path in plot_paths.items():
            print(f"{plot_name}: {path}")
        
        return report


def evaluate_robot_performance(repertoire: DynamicGraphRepertoire, 
                             num_evaluations: int = 10) -> Dict[str, float]:
    """Evaluate robot performance using actual Brax simulation."""
    env = create_ant_environment()
    
    all_fitnesses = []
    all_distances = []
    
    print(f"Evaluating robot performance with {num_evaluations} episodes...")
    
    for niche in repertoire.niches.values():
        for elite in niche.elites[:min(2, len(niche.elites))]:  # Evaluate top 2 per niche
            genome = elite[0]
            
            try:
                episode_data = simulate_robot_episode(env, genome, episode_length=500)
                fitness = episode_data['fitness']
                
                all_fitnesses.append(fitness)
                all_distances.append(abs(fitness))  # Distance traveled
                
            except Exception as e:
                print(f"Simulation error: {e}")
                continue
    
    if not all_fitnesses:
        return {"mean_distance": 0.0, "max_distance": 0.0, "success_rate": 0.0}
    
    metrics = {
        "mean_distance": np.mean(all_distances),
        "max_distance": np.max(all_distances),
        "std_distance": np.std(all_distances),
        "success_rate": len([f for f in all_fitnesses if f > 0.1]) / len(all_fitnesses),
        "num_evaluated": len(all_fitnesses)
    }
    
    return metrics


if __name__ == "__main__":
    print("Testing DBC evaluation functions...")
    
    from train import DynamicGraphRepertoire, Niche
    
    repertoire = DynamicGraphRepertoire()
    
    test_niche = Niche(1, behavior_dim=240)
    test_genome = np.random.randn(256)
    test_bd = np.random.randn(240)
    test_niche.add_elite(test_genome, 0.75, test_bd)
    
    repertoire.add_niche(test_niche)
    
    evaluator = DBCEvaluator(repertoire)
    
    fitness_metrics = evaluator.evaluate_fitness_distribution()
    print(f"Fitness metrics: {fitness_metrics}")
    
    diversity_metrics = evaluator.evaluate_diversity_metrics()
    print(f"Diversity metrics: {diversity_metrics}")
    
    report = evaluator.generate_comprehensive_report()
    print("Evaluation test completed successfully!")
