import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader, attack, device):
    """
    Evaluates the model for Clean Accuracy (CA) and Attack Success Rate (ASR).
    """
    model.eval()
    correct_clean, total_clean = 0, 0
    correct_poison, total_poison = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Clean Accuracy
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_clean += labels.size(0)
            correct_clean += (predicted == labels).sum().item()
            
            # Attack Success Rate
            if attack:
                non_target_mask = (labels != attack.target_class)
                if torch.any(non_target_mask):
                    poisoned_images = attack.apply_trigger(images[non_target_mask].clone())
                    poisoned_labels = torch.full_like(labels[non_target_mask], attack.target_class)
                    
                    outputs_poison = model(poisoned_images)
                    _, predicted_poison = torch.max(outputs_poison.data, 1)
                    total_poison += poisoned_labels.size(0)
                    correct_poison += (predicted_poison == poisoned_labels).sum().item()

    ca = 100 * correct_clean / total_clean if total_clean > 0 else 0
    asr = 100 * correct_poison / total_poison if total_poison > 0 else 0
    return ca, asr

def plot_training_curves(history, defense_name, condition, out_dir='.'):
    """
    Saves plots of training history (loss, CA, ASR) to a PDF file.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'Training Curves for {defense_name} ({condition})', fontsize=16)

    axes[0].plot(history['loss'], label='Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    
    axes[1].plot(history['ca'], label='Clean Accuracy', color='g')
    axes[1].set_title('Clean Accuracy (CA)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0, 101)

    axes[2].plot(history['asr'], label='Attack Success Rate', color='r')
    axes[2].set_title('Attack Success Rate (ASR)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Success Rate (%)')
    axes[2].set_ylim(0, 101)

    for ax in axes:
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(out_dir, f"training_curves_{defense_name.lower()}_{condition}.pdf")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(filename, format='pdf', bbox_inches="tight")
    print(f"Saved training curves to {filename}")
    plt.close()

def save_results_summary(results, condition, out_dir='.'):
    """
    Saves a JSON summary and a bar plot of the final experiment results.
    """
    filename = os.path.join(out_dir, f'experiment_summary_{condition}.json')
    os.makedirs(out_dir, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved summary of all results to {filename}")

    # Plot summary bar chart
    labels = list(results.keys())
    ca_values = [res['CA'] for res in results.values()]
    asr_values = [res['ASR'] for res in results.values()]

    x = np.arange(len(labels))
    width = 0.35

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    rects1 = ax.bar(x - width/2, ca_values, width, label='Clean Accuracy (CA)')
    rects2 = ax.bar(x + width/2, asr_values, width, label='Attack Success Rate (ASR)')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Comparison of Defense Methods ({condition})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--')

    fig.tight_layout()
    plot_filename = os.path.join(out_dir, f"accuracy_summary_{condition}.pdf")
    plt.savefig(plot_filename, format='pdf', bbox_inches="tight")
    print(f"Saved summary plot to {plot_filename}")
    plt.close()
