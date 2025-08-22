import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def get_embeddings(encoder, data, device):
    """Generate embeddings for the entire graph."""
    encoder.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index)
    return z

def evaluate_linear_probe(encoder, data, split_idx, device, seed=42):
    """Performs linear probing: trains a logistic regression classifier on frozen embeddings."""
    print(f"Running linear probe with seed {seed}...")
    embeddings = get_embeddings(encoder, data, device).cpu()
    
    train_idx = split_idx['train'].cpu()
    test_idx = split_idx['test'].cpu()
    labels = data.y.cpu()

    X_train = embeddings[train_idx]
    y_train = labels[train_idx].squeeze()
    X_test = embeddings[test_idx]
    y_test = labels[test_idx].squeeze()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = LogisticRegression(random_state=seed, max_iter=500, n_jobs=-1, C=0.1)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def evaluate_model_multiple_runs(encoder, data, split_idx, device, num_runs=5):
    """Evaluate model with multiple runs for statistical reliability."""
    accuracies = []
    for i in range(num_runs):
        acc = evaluate_linear_probe(encoder, data, split_idx, device, seed=i)
        accuracies.append(acc)
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"Evaluation finished. Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    return mean_acc, std_acc, accuracies
