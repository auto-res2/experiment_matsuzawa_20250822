import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, test_loader, device):
    """
    Evaluates the model on the test set.
    """
    model.eval()
    total_correct = 0.0
    total_samples = 0.0
    for img_seq, label in tqdm(test_loader, desc="Evaluating", leave=False):
        img_seq, label = img_seq.to(device, non_blocking=True), label.to(device, non_blocking=True)
        output_seq = model(img_seq)
        output_potential = output_seq.mean(dim=0)
        total_correct += (output_potential.argmax(dim=1) == label).sum().item()
        total_samples += label.size(0)
    return total_correct / total_samples if total_samples > 0 else 0.0
