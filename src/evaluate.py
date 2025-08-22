import math
import os
import random
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import (
    set_seed, SimpleTokenizer, ForwardPairsDataset, 
    collate_forward
)
from train import (
    MockBaseLM, AdapterEntityRelHead, mix_logits, 
    get_topk_with_gold, TrainConfig
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_pdf_plot(fig, filename: str):
    """Save matplotlib figure as PDF."""
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    fig.savefig(filename, bbox_inches="tight", format='pdf', dpi=300)
    plt.close(fig)

def evaluate_forward_reverse(base_model, adapter, tokenizer, entity_ids, 
                           forward_pairs, reverse_pairs, config: TrainConfig,
                           save_dir: str = ".research/iteration1/images"):
    """Evaluate forward and reverse accuracy."""
    os.makedirs(save_dir, exist_ok=True)
    
    forward_dataset = ForwardPairsDataset(forward_pairs, config.relation_names[0])
    forward_loader = DataLoader(forward_dataset, batch_size=config.batch_size, shuffle=False,
                               collate_fn=lambda batch: collate_forward(batch, tokenizer))
    
    reverse_rel = config.relation_names[0].replace("->", "<-") if "->" in config.relation_names[0] else "<-R1"
    reverse_dataset = ForwardPairsDataset(reverse_pairs, reverse_rel)
    reverse_loader = DataLoader(reverse_dataset, batch_size=config.batch_size, shuffle=False,
                               collate_fn=lambda batch: collate_forward(batch, tokenizer))
    
    base_model.eval()
    adapter.eval()
    
    def evaluate_loader(loader, rel_name, description):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                target_ids = batch['target_ids'].to(DEVICE)
                rel = batch['rel']
                
                base_output = base_model(input_ids, output_hidden_states=True)
                base_logits = base_output.logits[:, -1, :]
                h_t = base_output.hidden_states[0][:, -1, :]
                
                candidate_ids, gold_pos = get_topk_with_gold(base_logits, target_ids, config.K)
                
                entity_candidates = []
                for batch_candidates in candidate_ids:
                    batch_entity_cands = []
                    for cand_id in batch_candidates:
                        if cand_id.item() in entity_ids:
                            batch_entity_cands.append(cand_id.item())
                    if len(batch_entity_cands) == 0:
                        batch_entity_cands = [entity_ids[0]]
                    entity_candidates.extend(batch_entity_cands)
                
                entity_candidates = torch.tensor(entity_candidates, device=candidate_ids.device, dtype=torch.long)
                l_adapter, alpha = adapter(h_t, rel_name, entity_candidates)
                
                n_batch, n_cands = candidate_ids.size()
                if l_adapter.size(0) == n_batch:
                    if l_adapter.size(1) >= n_cands:
                        l_adapter = l_adapter[:, :n_cands]
                    else:
                        pad_size = n_cands - l_adapter.size(1)
                        padding = torch.zeros(n_batch, pad_size, device=l_adapter.device)
                        l_adapter = torch.cat([l_adapter, padding], dim=1)
                else:
                    l_adapter = torch.zeros(n_batch, n_cands, device=l_adapter.device)
                
                l_base_sub = torch.gather(base_logits, 1, candidate_ids)
                l_mix = mix_logits(l_base_sub, l_adapter, alpha)
                
                pred_pos = torch.argmax(l_mix, dim=1)
                correct += (pred_pos == gold_pos).sum().item()
                total += len(pred_pos)
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"{description} Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    forward_acc = evaluate_loader(forward_loader, config.relation_names[0], "Forward")
    reverse_acc = evaluate_loader(reverse_loader, config.relation_names[0], "Reverse")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    categories = ['Forward', 'Reverse']
    accuracies = [forward_acc, reverse_acc]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax.bar(categories, accuracies, color=colors, alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('REA-Cycle++ Forward vs Reverse Accuracy')
    ax.set_ylim(0, 1.0)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_pdf_plot(fig, os.path.join(save_dir, "forward_reverse_accuracy.pdf"))
    
    return {
        'forward_accuracy': forward_acc,
        'reverse_accuracy': reverse_acc,
        'reversal_gap': forward_acc - reverse_acc
    }

def evaluate_composition(base_model, adapter, tokenizer, entity_ids, chains, config: TrainConfig,
                        save_dir: str = ".research/iteration1/images"):
    """Evaluate composition/transitivity."""
    os.makedirs(save_dir, exist_ok=True)
    
    base_model.eval()
    adapter.eval()
    
    composition_errors = []
    cycle_errors = []
    
    with torch.no_grad():
        eval_chains = chains[:min(50, len(chains))]
        
        for a, b, c in eval_chains:
            a_id = tokenizer.convert_tokens_to_ids(a)
            b_id = tokenizer.convert_tokens_to_ids(b)
            c_id = tokenizer.convert_tokens_to_ids(c)
            
            a_idx = entity_ids.index(a_id) if a_id in entity_ids else 0
            b_idx = entity_ids.index(b_id) if b_id in entity_ids else 0
            c_idx = entity_ids.index(c_id) if c_id in entity_ids else 0
            
            E_a = adapter.E[a_idx:a_idx+1]  # (1, k)
            E_b = adapter.E[b_idx:b_idx+1]
            E_c = adapter.E[c_idx:c_idx+1]
            
            M = adapter.orthogonal_core(config.relation_names[0])
            
            z_ab = E_a @ M.T
            z_abc = z_ab @ M.T
            comp_error = torch.norm(z_abc - E_c).item()
            composition_errors.append(comp_error)
            
            M_inv = M.T  # For orthogonal matrices
            cycle_matrix = M_inv @ M
            I = torch.eye(M.size(0), device=M.device)
            cycle_error = torch.norm(cycle_matrix - I, p='fro').item()
            cycle_errors.append(cycle_error)
    
    avg_comp_error = np.mean(composition_errors)
    avg_cycle_error = np.mean(cycle_errors)
    
    print(f"Average Composition Error: {avg_comp_error:.6f}")
    print(f"Average Cycle Error: {avg_cycle_error:.6f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(composition_errors, bins=20, alpha=0.7, color='#F18F01')
    ax1.set_xlabel('Composition Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Composition Errors')
    ax1.axvline(avg_comp_error, color='red', linestyle='--', label=f'Mean: {avg_comp_error:.4f}')
    ax1.legend()
    
    ax2.hist(cycle_errors, bins=20, alpha=0.7, color='#C73E1D')
    ax2.set_xlabel('Cycle Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Cycle Errors')
    ax2.axvline(avg_cycle_error, color='red', linestyle='--', label=f'Mean: {avg_cycle_error:.4f}')
    ax2.legend()
    
    plt.tight_layout()
    save_pdf_plot(fig, os.path.join(save_dir, "composition_cycle_errors.pdf"))
    
    return {
        'composition_errors': composition_errors,
        'cycle_errors': cycle_errors,
        'avg_composition_error': avg_comp_error,
        'avg_cycle_error': avg_cycle_error
    }

def plot_training_curves(losses: List[float], save_dir: str = ".research/iteration1/images"):
    """Plot training loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, 'o-', color='#3A86FF', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('REA-Cycle++ Training Loss')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_pdf_plot(fig, os.path.join(save_dir, "training_loss.pdf"))

def evaluate_orthogonality(adapter, config: TrainConfig, save_dir: str = ".research/iteration1/images"):
    """Evaluate orthogonality of relation matrices."""
    os.makedirs(save_dir, exist_ok=True)
    
    adapter.eval()
    
    with torch.no_grad():
        ortho_errors = []
        spectral_norms = []
        
        for rel in config.relation_names:
            M = adapter.orthogonal_core(rel)
            
            I = torch.eye(M.size(0), device=M.device)
            ortho_error = torch.norm(M.T @ M - I, p='fro').item()
            ortho_errors.append(ortho_error)
            
            spectral_norm = torch.linalg.matrix_norm(M, ord=2).item()
            spectral_norms.append(spectral_norm)
    
    print(f"Orthogonality Errors: {ortho_errors}")
    print(f"Spectral Norms: {spectral_norms}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    relations = [f"Rel {i+1}" for i in range(len(config.relation_names))]
    
    ax1.bar(relations, ortho_errors, color='#FF006E', alpha=0.7)
    ax1.set_ylabel('Orthogonality Error')
    ax1.set_title('Orthogonality Error by Relation')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(relations, spectral_norms, color='#8338EC', alpha=0.7)
    ax2.set_ylabel('Spectral Norm')
    ax2.set_title('Spectral Norm by Relation')
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Ideal (1.0)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_pdf_plot(fig, os.path.join(save_dir, "orthogonality_metrics.pdf"))
    
    return {
        'orthogonality_errors': ortho_errors,
        'spectral_norms': spectral_norms
    }

def comprehensive_evaluation(results_dict, config: TrainConfig, save_dir: str = ".research/iteration1/images"):
    """Run comprehensive evaluation of REA-Cycle++ model."""
    print("\n" + "="*50)
    print("COMPREHENSIVE EVALUATION OF REA-CYCLE++")
    print("="*50)
    
    base_model = results_dict['base_model']
    adapter = results_dict['adapter']
    tokenizer = results_dict['tokenizer']
    entity_ids = results_dict['entity_ids']
    forward_pairs = results_dict['forward_pairs']
    reverse_pairs = results_dict['reverse_pairs']
    chains = results_dict['chains']
    losses = results_dict['losses']
    
    print("\n1. Plotting training curves...")
    plot_training_curves(losses, save_dir)
    
    print("\n2. Evaluating forward vs reverse accuracy...")
    accuracy_results = evaluate_forward_reverse(
        base_model, adapter, tokenizer, entity_ids,
        forward_pairs, reverse_pairs, config, save_dir
    )
    
    print("\n3. Evaluating composition and cycle consistency...")
    composition_results = evaluate_composition(
        base_model, adapter, tokenizer, entity_ids,
        chains, config, save_dir
    )
    
    print("\n4. Evaluating orthogonality of relation matrices...")
    orthogonality_results = evaluate_orthogonality(adapter, config, save_dir)
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Forward Accuracy: {accuracy_results['forward_accuracy']:.4f}")
    print(f"Reverse Accuracy: {accuracy_results['reverse_accuracy']:.4f}")
    print(f"Reversal Gap: {accuracy_results['reversal_gap']:.4f}")
    print(f"Avg Composition Error: {composition_results['avg_composition_error']:.6f}")
    print(f"Avg Cycle Error: {composition_results['avg_cycle_error']:.6f}")
    print(f"Avg Orthogonality Error: {np.mean(orthogonality_results['orthogonality_errors']):.6f}")
    print(f"Avg Spectral Norm: {np.mean(orthogonality_results['spectral_norms']):.4f}")
    
    return {
        'accuracy': accuracy_results,
        'composition': composition_results,
        'orthogonality': orthogonality_results
    }

if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
