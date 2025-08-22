import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_model(model, h5_path, device, eval_batch_size, stream_h5_data_func):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data_batch in stream_h5_data_func(h5_path, 'test', eval_batch_size):
            coords = torch.from_numpy(data_batch['coords']).to(device)
            normals = torch.from_numpy(data_batch['normals']).to(device)
            labels = torch.from_numpy(data_batch['labels']).to(device)
            features = torch.cat([coords, normals], dim=1)
            
            logits = model(features)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    cm = confusion_matrix(all_labels, all_preds)
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-9)
    miou = np.nanmean(iou)
    oa = accuracy_score(all_labels, all_preds)
    return miou, oa, cm
