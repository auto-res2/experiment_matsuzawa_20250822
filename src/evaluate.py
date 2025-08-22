import torch
import numpy as np
from typing import Tuple

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dl) -> Tuple[float, float]:
    model.eval()
    losses = []
    total = 0
    correct = 0
    for batch in dl:
        inp = batch['input_ids'].to(next(model.parameters()).device)
        labels = batch['labels'].to(next(model.parameters()).device)
        logits, loss = model(inp, labels=labels)
        losses.append(loss.item())
        preds = logits.argmax(-1)
        total += labels.numel()
        correct += (preds == labels).sum().item()
    loss = float(np.mean(losses)) if losses else float('nan')
    acc = correct / max(1, total)
    return loss, acc
