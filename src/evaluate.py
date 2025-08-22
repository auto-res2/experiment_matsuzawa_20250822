import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def evaluate_linear_one_epoch(backbone, classifier, dataloader, criterion, config):
    backbone.eval()
    classifier.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(config['DEVICE']), labels.to(config['DEVICE'])
            features = backbone(images, is_pretrain=False)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy

def plot_results(results, config):
    print("\n--- Generating and Saving Plots ---")
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.figsize'] = (8, 5)
    
    images_dir = os.path.join(config['OUTPUT_DIR'], "images")
    os.makedirs(images_dir, exist_ok=True)

    # Plot 1: Pre-training Loss
    plt.figure()
    for method, res in results.items():
        if 'pretrain_loss' in res:
            plt.plot(res['pretrain_loss'], label=method.upper())
    plt.title('Pre-training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(images_dir, "pretraining_loss.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved pretraining_loss.pdf to {images_dir}")

    # Plot 2: Linear Evaluation Accuracy
    plt.figure()
    for method, res in results.items():
        if 'linear_eval' in res:
            plt.plot(res['linear_eval']['val_acc'], label=method.upper())
    plt.title('Linear Evaluation Accuracy on Test Set')
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(images_dir, "linear_eval_accuracy.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved linear_eval_accuracy.pdf to {images_dir}")

    # Plot 3: Final Accuracy Comparison Bar Chart
    final_accuracies = {m.upper(): r['linear_eval']['final_acc'] for m, r in results.items() if 'linear_eval' in r}
    if not final_accuracies:
        print("No final accuracies to plot.")
        return

    df = pd.DataFrame(list(final_accuracies.items()), columns=['Method', 'Accuracy']).sort_values('Accuracy', ascending=False)
    
    plt.figure()
    ax = sns.barplot(x='Method', y='Accuracy', data=df, palette='viridis')
    ax.bar_label(ax.containers[0], fmt='%.2f')
    plt.title('Final Top-1 Test Accuracy Comparison')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.ylim(0, max(df['Accuracy']) * 1.2 if not df.empty else 100)
    plt.savefig(os.path.join(images_dir, "final_accuracy_comparison.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved final_accuracy_comparison.pdf to {images_dir}")

def print_summary(results):
    print("\n--- Final Experiment Results ---")
    summary = []
    for method, res in results.items():
        if 'linear_eval' in res:
            summary.append({'Method': method.upper(), 'Final Accuracy': f"{res['linear_eval']['final_acc']:.2f}%"})
    
    if not summary:
        print("No results to display.")
        return

    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
