import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.cluster import KMeans
from scipy.stats import entropy
from rouge_score import rouge_scorer
from tslearn.barycenters import dtw_barycenter_averaging
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='tslearn')

def calculate_ltd_score(trajectories: List[np.ndarray]) -> float:
    """Calculates the Latent Trajectory Divergence (LTD) score."""
    if len(trajectories) < 2:
        return 0.0
    
    formatted_trajectories = [t.astype(np.float64) for t in trajectories if t.shape[0] > 0]
    if len(formatted_trajectories) < 2:
        return 0.0
    
    try:
        barycenter = dtw_barycenter_averaging(formatted_trajectories, max_iter=5, tol=1e-3)
    except Exception as e:
        print(f"Error in DBA: {e}. Returning 0.0")
        return 0.0

    total_divergence = 0.0
    for traj in formatted_trajectories:
        distance, _ = fastdtw(traj, barycenter, dist=cosine)
        total_divergence += distance
    
    return total_divergence / len(formatted_trajectories)

def calculate_semantic_entropy(texts: List[str], sbert_model: SentenceTransformer, k_se: int = 3) -> float:
    """Calculates Semantic Entropy (SE)."""
    if len(texts) < k_se:
        return 0.0
    embeddings = sbert_model.encode(texts, show_progress_bar=False)
    kmeans = KMeans(n_clusters=k_se, random_state=42, n_init='auto').fit(embeddings)
    cluster_distribution = np.bincount(kmeans.labels_) / len(kmeans.labels_)
    return float(entropy(cluster_distribution))

def calculate_lexical_similarity(texts: List[str]) -> float:
    """Calculates average pairwise ROUGE-L F1 score."""
    if len(texts) < 2:
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    pair_scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = scorer.score(texts[i], texts[j])['rougeL'].fmeasure
            pair_scores.append(score)
    return float(1.0 - np.mean(pair_scores))

def calculate_haloscope_score_mock(texts: List[str]) -> float:
    """Mock function for HaloScope baseline."""
    scores = [len(text) * (1 + 0.1 * sum(c.isdigit() for c in text)) for text in texts]
    return float(np.mean(scores))

def label_generation(generated_text: str, ground_truth_answers: List[str]) -> int:
    """Labels a generation as Correct (1) or Incorrect (0)."""
    text_lower = generated_text.lower()
    for answer in ground_truth_answers:
        if answer in text_lower:
            return 1
    return 0

def evaluate_uncertainty_methods(results_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all uncertainty quantification methods."""
    uq_methods = {
        'LTD': 'ltd_score',
        'Semantic Entropy': 'se_score', 
        'Lexical Diversity': 'lex_div_score',
        'HaloScope (Mock)': 'halo_score'
    }
    
    eval_summary = []
    y_true = 1 - results_df['label']

    for name, col in uq_methods.items():
        y_score = results_df[col]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = auc(fpr, tpr)
        
        auprc = average_precision_score(y_true, y_score)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = (2 * precision * recall) / (precision + recall)
        best_f1 = np.nanmax(f1_scores)
        
        eval_summary.append({
            'Method': name, 
            'AUROC': auroc, 
            'AUPRC': auprc, 
            'Best F1': best_f1
        })

    return pd.DataFrame(eval_summary)

def plot_roc_curves(results_df: pd.DataFrame, model_name_str: str, save_dir: str = ".research/iteration1/images"):
    """Plot and save ROC curves for all UQ methods."""
    plt.figure(figsize=(10, 8))
    
    uq_methods = {
        'LTD': 'ltd_score',
        'Semantic Entropy': 'se_score',
        'Lexical Diversity': 'lex_div_score', 
        'HaloScope (Mock)': 'halo_score'
    }

    for name, col in uq_methods.items():
        y_true = 1 - results_df['label']
        y_score = results_df[col]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUROC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_name_str}')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"roc_curves_comparison_{model_name_str.replace('/','_')}.pdf")
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved ROC curve plot to {filename}")
    plt.close()

def plot_score_distributions(results_df: pd.DataFrame, model_name_str: str, save_dir: str = ".research/iteration1/images"):
    """Plot and save score distributions for each method."""
    uq_methods = {
        'LTD': 'ltd_score',
        'Semantic Entropy': 'se_score',
        'Lexical Diversity': 'lex_div_score',
        'HaloScope (Mock)': 'halo_score'
    }
    results_df['Answer Type'] = results_df['label'].apply(lambda x: 'Correct' if x == 1 else 'Incorrect')

    os.makedirs(save_dir, exist_ok=True)
    for name, col in uq_methods.items():
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=results_df, x='Answer Type', y=col, inner='quartile')
        plt.title(f'Score Distribution for {name} - {model_name_str}')
        plt.ylabel(f'{name} Score')
        plt.xlabel('Ground Truth Answer Type')
        
        filename = os.path.join(save_dir, f"score_distributions_{name.lower().replace(' ', '_')}_{model_name_str.replace('/','_')}.pdf")
        plt.savefig(filename, bbox_inches="tight")
        print(f"Saved distribution plot to {filename}")
        plt.close()

if __name__ == "__main__":
    print("=== Model Evaluation Module ===")
    
    mock_results = pd.DataFrame({
        'question_id': ['test1', 'test2'] * 5,
        'ltd_score': np.random.rand(10),
        'se_score': np.random.rand(10),
        'lex_div_score': np.random.rand(10),
        'halo_score': np.random.rand(10),
        'label': [1, 0] * 5
    })
    
    eval_summary = evaluate_uncertainty_methods(mock_results)
    print("Evaluation summary:")
    print(eval_summary.to_string(index=False))
    
    plot_roc_curves(mock_results, "test_model")
    plot_score_distributions(mock_results, "test_model")
    
    print("Evaluation module working correctly!")
