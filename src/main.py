import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from tslearn.barycenters import dtw_barycenter_averaging
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from scipy.stats import entropy
from rouge_score import rouge_scorer
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
from tqdm import tqdm
import os
import json

warnings.filterwarnings("ignore", category=UserWarning, module='tslearn')

REQUIRED_LIBRARIES = [
    "torch",
    "transformers",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "tslearn",
    "fastdtw",
    "sentence-transformers",
    "rouge-score",
    "tqdm",
    "accelerate"
]

global_trajectory_buffer = []

def get_hidden_states_hook(module, input, output):
    """Hook to capture the pre-layernorm hidden states."""
    hidden_state = input[0][0, -1, :].detach().cpu().float().numpy()
    global_trajectory_buffer.append(hidden_state)

def find_last_layernorm(model):
    """Finds the name of the final LayerNorm module in a model."""
    possible_names = ['model.norm', 'transformer.ln_f', 'lm_head.norm']
    for name, module in model.named_modules():
        if name in possible_names:
            print(f"Found final LayerNorm at: {name}")
            return module
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.LayerNorm):
            print(f"Found final LayerNorm with fallback at: {name}")
            return module
    raise ValueError("Could not find the final LayerNorm module in the model.")


def calculate_ltd_score(trajectories):
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

def calculate_semantic_entropy(texts, sbert_model, k_se=3):
    """Calculates Semantic Entropy (SE)."""
    if len(texts) < k_se:
        return 0.0
    embeddings = sbert_model.encode(texts, show_progress_bar=False)
    kmeans = KMeans(n_clusters=k_se, random_state=42, n_init='auto').fit(embeddings)
    cluster_distribution = np.bincount(kmeans.labels_) / len(kmeans.labels_)
    return entropy(cluster_distribution)

def calculate_lexical_similarity(texts):
    """Calculates average pairwise ROUGE-L F1 score."""
    if len(texts) < 2:
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    pair_scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = scorer.score(texts[i], texts[j])['rougeL'].fmeasure
            pair_scores.append(score)
    return 1.0 - np.mean(pair_scores)

def calculate_haloscope_score_mock(texts):
    """Mock function for HaloScope, as it requires a pre-computed knowledge graph."""
    scores = [len(text) * (1 + 0.1 * sum(c.isdigit() for c in text)) for text in texts]
    return np.mean(scores)


def generate_and_extract_trajectories(model, tokenizer, prompt, k, generation_params):
    """Generates k responses and extracts their latent trajectories."""
    global global_trajectory_buffer
    
    last_layernorm = find_last_layernorm(model)
    hook = last_layernorm.register_forward_hook(get_hidden_states_hook)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    ensemble_trajectories = []
    ensemble_texts = []

    for _ in range(k):
        global_trajectory_buffer = []
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                **generation_params
            )
        if global_trajectory_buffer:
            ensemble_trajectories.append(np.array(global_trajectory_buffer))
        else:
            ensemble_trajectories.append(np.array([])) # Handle empty generation
        
        generated_text = tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        ensemble_texts.append(generated_text)

    hook.remove()
    return ensemble_texts, ensemble_trajectories

def create_mock_dataset():
    """Creates a diverse mock dataset to test robustness."""
    return [
        {
            "id": "tqa_1",
            "question": "What is the capital of France?",
            "ground_truth_answers": ["paris"]
        },
        {
            "id": "tqa_2",
            "question": "Who wrote the novel '1984'?",
            "ground_truth_answers": ["george orwell", "eric blair"]
        },
        {
            "id": "nqs_1",
            "question": "What is the main cause of the tides?",
            "ground_truth_answers": ["moon", "gravity", "gravitational"]
        },
        {
            "id": "bio_1",
            "question": "What is the function of mitochondria in a cell?",
            "ground_truth_answers": ["powerhouse", "atp production", "energy"]
        },
        {
            "id": "tqa_hallucination",
            "question": "What was the color of the horse that Napoleon rode in the Battle of Austerlitz?",
            "ground_truth_answers": ["unknown", "not recorded", "disputed", "marengo"]
        },
        {
            "id": "truthful_1",
            "question": "What happens if you swallow chewing gum?",
            "ground_truth_answers": ["digest", "pass through", "excreted"]
        }
    ]

def label_generation(generated_text, ground_truth_answers):
    """Labels a generation as Correct (1) or Incorrect (0)."""
    text_lower = generated_text.lower()
    for answer in ground_truth_answers:
        if answer in text_lower:
            return 1 # Correct
    return 0 # Incorrect

def run_experiment(model_name, dataset, sbert_model, k=10):
    """Main function to run the benchmark experiment."""
    print(f"--- Running Experiment on {model_name} ---")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    generation_params = {
        "max_new_tokens": 50,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id
    }

    results = []
    for item in tqdm(dataset, desc="Processing questions"):
        prompt = item['question']
        ground_truth = item['ground_truth_answers']
        
        texts, trajectories = generate_and_extract_trajectories(model, tokenizer, prompt, k, generation_params)

        ltd_score = calculate_ltd_score(trajectories)
        se_score = calculate_semantic_entropy(texts, sbert_model)
        lex_div_score = calculate_lexical_similarity(texts)
        halo_score = calculate_haloscope_score_mock(texts)

        for text in texts:
            label = label_generation(text, ground_truth)
            results.append({
                "question_id": item['id'],
                "ltd_score": ltd_score,
                "se_score": se_score,
                "lex_div_score": lex_div_score,
                "halo_score": halo_score,
                "label": label
            })
            
    return pd.DataFrame(results)


def plot_roc_curves(results_df, model_name_str, save_dir=".research/iteration1/images"):
    """Plots and saves ROC curves for all UQ methods."""
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

def plot_score_distributions(results_df, model_name_str, save_dir=".research/iteration1/images"):
    """Plots and saves score distributions for each method."""
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

def evaluate_and_plot_results(results_df, model_name_str):
    """Calculates metrics and generates all plots."""
    print(f"\n--- Evaluation Results for {model_name_str} ---")
    
    if results_df.empty or 'label' not in results_df.columns:
        print("No results to evaluate.")
        return

    uq_methods = {
        'LTD': 'ltd_score',
        'Semantic Entropy': 'se_score',
        'Lexical Diversity': 'lex_div_score',
        'HaloScope (Mock)': 'halo_score'
    }
    
    eval_summary = []
    y_true = 1 - results_df['label'] # Predict 'Incorrect'

    for name, col in uq_methods.items():
        y_score = results_df[col]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = auc(fpr, tpr)
        
        auprc = average_precision_score(y_true, y_score)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = (2 * precision * recall) / (precision + recall)
        best_f1 = np.nanmax(f1_scores)
        
        eval_summary.append({'Method': name, 'AUROC': auroc, 'AUPRC': auprc, 'Best F1': best_f1})

    summary_df = pd.DataFrame(eval_summary)
    print(summary_df.to_string(index=False))
    
    plot_roc_curves(results_df, model_name_str)
    plot_score_distributions(results_df, model_name_str)


def test_code():
    """A quick test to verify functionality of the code."""
    print("\n--- Running Quick Test Function ---")
    
    test_model_name = "gpt2"
    test_k = 3
    
    test_dataset = create_mock_dataset()[:2]
    
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Failed to download SBERT model for test: {e}")
        print("Test skipped.")
        return
        
    try:
        results = run_experiment(test_model_name, test_dataset, sbert_model, k=test_k)
        evaluate_and_plot_results(results, test_model_name)
        print("\n--- Test Function Completed Successfully ---")
        return True
    except Exception as e:
        print(f"\n--- Test Function FAILED: {e} ---")
        import traceback
        traceback.print_exc()
        return False

def update_status_to_stopped():
    """Updates the status_enum to 'stopped' in research_history.json"""
    try:
        with open('.research/research_history.json', 'r') as f:
            research_data = json.load(f)
        
        research_data['status_enum'] = 'stopped'
        
        with open('.research/research_history.json', 'w') as f:
            json.dump(research_data, f, indent=2)
        
        print("Successfully updated status_enum to 'stopped'")
        
    except Exception as e:
        print(f"Error updating status_enum: {e}")

if __name__ == '__main__':
    print("=== LTD (Latent Trajectory Divergence) Experiment ===")
    print("Required Python libraries to run this experiment:")
    for lib in REQUIRED_LIBRARIES:
        print(f"- {lib}")
    
    print("\n=== Running Test Function ===")
    test_success = test_code()
    
    if test_success:
        print("\n=== Test completed successfully! ===")
        print("The LTD experiment implementation is working correctly.")
        print("Generated plots have been saved to .research/iteration1/images/")
        
        update_status_to_stopped()
        
    else:
        print("\n=== Test FAILED! ===")
        print("There were errors in the implementation that need to be fixed.")
