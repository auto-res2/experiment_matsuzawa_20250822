import os
import torch
from torch.utils.data import Dataset
import torch_fidelity
import matplotlib.pyplot as plt
import seaborn as sns

class ImageDatasetWrapper(Dataset):
    """Wrapper for torch-fidelity to read from a tensor."""
    def __init__(self, images_tensor):
        # torch-fidelity expects uint8 images in [0, 255]
        self.images = images_tensor.mul(255).clamp(0, 255).to(torch.uint8)
        # It also expects images in HWC format
        self.images = self.images.permute(0, 2, 3, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], 0

def calculate_fid(generated_dataset, method_name, config):
    """Calculates the FID score for a generated dataset."""
    print(f"Calculating FID for {method_name}...")
    # Images are expected to be in [-1, 1], shift to [0, 1] for the wrapper
    images_for_fid = (generated_dataset.tensors[0] + 1.0) / 2.0
    wrapped_dataset = ImageDatasetWrapper(images_for_fid)
    
    try:
        metrics = torch_fidelity.calculate_metrics(
            input1=wrapped_dataset,
            input2='cifar10-train',
            cuda=torch.cuda.is_available(),
            fid=True,
            verbose=False
        )
        fid_score = metrics['frechet_inception_distance']
        print(f"FID Score for {method_name}: {fid_score:.2f}")
        return fid_score
    except Exception as e:
        print(f"Could not calculate FID: {e}. Returning NaN.")
        return float('nan')

def plot_training_curves(history, condition_name, config):
    """Plots and saves training loss and test accuracy curves."""
    if history is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.set_title(f'Training Loss ({condition_name.replace("_", " ").title()})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, history['test_acc'], 'ro-', label='Test Accuracy')
    ax2.set_title(f'Test Accuracy ({condition_name.replace("_", " ").title()})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()

    fig.suptitle(f'Training Dynamics on {condition_name.replace("_", " ").title()} Data', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(config['OUTPUT_DIR'], f'training_curves_{condition_name}.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close(fig)

def plot_summary_results(results_df, config):
    """Plots and saves the final summary bar charts."""
    # Reorder for plotting
    method_order = ['Real Data', 'DistDiff', 'DCD-PCE', 'Real-Fake', 'Standard Diffusion']
    results_df = results_df.reindex(method_order).dropna(how='all')

    sns.set_theme(style="whitegrid")

    # Accuracy Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(results_df.index, results_df['Accuracy'], yerr=results_df['Accuracy Std'], capsize=5, color=sns.color_palette("viridis", len(results_df)))
    ax.set_ylabel('Downstream Accuracy (%)', fontsize=12)
    ax.set_title('Downstream Classifier Accuracy on Real CIFAR-10 Test Set', fontsize=14, weight='bold')
    ax.bar_label(bars, fmt='%.2f')
    plt.xticks(rotation=15, ha='right')
    plt.ylim(bottom=0, top=max(100, results_df['Accuracy'].max() * 1.1))
    plt.tight_layout()
    save_path = os.path.join(config['OUTPUT_DIR'], 'accuracy_comparison.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved accuracy plot to {save_path}")
    plt.close(fig)

    # Generation Time Plot (excluding Real Data)
    gen_df = results_df[results_df['Generation Time (ms/image)'] > 0]
    if not gen_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(gen_df.index, gen_df['Generation Time (ms/image)'], color=sns.color_palette("magma", len(gen_df)))
        ax.set_ylabel('Time per Image (ms)', fontsize=12)
        ax.set_title('Data Generation Speed', fontsize=14, weight='bold')
        ax.set_yscale('log')
        ax.bar_label(bars, fmt='%.2f')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        save_path = os.path.join(config['OUTPUT_DIR'], 'generation_time_comparison.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved generation time plot to {save_path}")
        plt.close(fig)

    # FID Score Plot (excluding Real Data)
    fid_df = results_df[results_df['FID'] > 0]
    if not fid_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(fid_df.index, fid_df['FID'], color=sns.color_palette("plasma", len(fid_df)))
        ax.set_ylabel('Fréchet Inception Distance (FID)', fontsize=12)
        ax.set_title('Perceptual Quality (Lower is Better)', fontsize=14, weight='bold')
        ax.bar_label(bars, fmt='%.2f')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        save_path = os.path.join(config['OUTPUT_DIR'], 'fid_comparison.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved FID plot to {save_path}")
        plt.close(fig)
