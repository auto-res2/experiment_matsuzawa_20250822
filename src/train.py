import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import numpy as np
from tqdm import tqdm

def train_classifier(train_dataset, test_loader, method_name, seed, config):
    """Trains a downstream classifier on a given dataset."""
    print(f"\n--- Training downstream classifier on data from '{method_name}' (Seed {seed+1}/{config['NUM_SEEDS']}) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = resnet18(weights=None, num_classes=config['NUM_CLASSES'])
    model.to(config['DEVICE'])

    train_loader = DataLoader(train_dataset, batch_size=config['TRAIN_BATCH_SIZE'], shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['TRAIN_EPOCHS'])

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(config['TRAIN_EPOCHS']):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['TRAIN_EPOCHS']}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(config['DEVICE']), labels.to(config['DEVICE'])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(config['DEVICE']), labels.to(config['DEVICE'])
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        history['test_acc'].append(accuracy)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Accuracy = {accuracy:.2f}%")

    final_accuracy = history['test_acc'][-1]
    print(f"Final Test Accuracy for '{method_name}' (Seed {seed+1}): {final_accuracy:.2f}%")
    return final_accuracy, history
