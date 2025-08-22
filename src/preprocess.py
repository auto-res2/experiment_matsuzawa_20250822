import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.cluster import MiniBatchKMeans

def create_mock_dales_h5(path, num_train, num_test, num_scenes, num_classes):
    """Generates a mock HDF5 file mimicking the DALES dataset structure."""
    if os.path.exists(path):
        print(f"Mock dataset '{path}' already exists. Skipping generation.")
        return
    print(f"Generating mock dataset at '{path}'...")
    with h5py.File(path, 'w') as f:
        for split in ['train', 'test']:
            f.create_group(split)
            num_points_split = num_train if split == 'train' else num_test
            points_per_scene = num_points_split // num_scenes
            for i in range(num_scenes):
                scene_group = f.create_group(f'{split}/scene_{i}')
                
                coords, labels = [], []
                plane_pts = int(points_per_scene * 0.4)
                c1 = np.random.rand(plane_pts, 3) * np.array([100, 100, 0.5])
                l1 = np.full(plane_pts, 0)
                coords.append(c1); labels.append(l1)
                
                building_pts = int(points_per_scene * 0.3)
                c2 = np.random.rand(building_pts, 3) * np.array([20, 20, 50]) + np.array([40, 40, 0])
                l2 = np.full(building_pts, 1)
                coords.append(c2); labels.append(l2)

                veg_pts = int(points_per_scene * 0.2)
                for _ in range(5):
                    center = np.random.rand(3) * np.array([100, 100, 30])
                    radius = np.random.rand() * 5 + 2
                    phi = 2 * np.pi * np.random.rand(veg_pts // 5)
                    costheta = 2 * np.random.rand(veg_pts // 5) - 1
                    theta = np.arccos(costheta)
                    x = radius * np.sin(theta) * np.cos(phi)
                    y = radius * np.sin(theta) * np.sin(phi)
                    z = radius * np.cos(theta)
                    c3_part = np.vstack([x,y,z]).T + center
                    coords.append(c3_part)
                l3 = np.full(sum(c.shape[0] for c in coords[-5:]), 2)
                labels.append(l3)

                line_pts = points_per_scene - sum(c.shape[0] for c in coords)
                start, end = np.random.rand(2, 3) * np.array([100, 100, 60])
                c4 = start + (end - start) * np.random.rand(line_pts, 1)
                l4 = np.full(line_pts, 3)
                coords.append(c4); labels.append(l4)

                all_coords = np.concatenate(coords, axis=0)
                all_labels = np.concatenate(labels, axis=0)
                
                all_coords -= all_coords.mean(axis=0)
                all_coords /= np.abs(all_coords).max()

                scene_group.create_dataset('coords', data=all_coords.astype(np.float32))
                scene_group.create_dataset('labels', data=all_labels.astype(np.int64))
                scene_group.create_dataset('features', data=np.random.rand(all_coords.shape[0], 3).astype(np.float32))
    print("Mock dataset generation complete.")

def compute_and_store_normals(h5_path):
    print("Computing and storing normals (mock)... ")
    with h5py.File(h5_path, 'a') as f:
        for split in ['train', 'test']:
            for name, group in f[split].items():
                if 'normals' in group:
                    continue
                num_points = group['coords'].shape[0]
                normals = np.random.randn(num_points, 3).astype(np.float32)
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                group.create_dataset('normals', data=normals)
    print("Normals computed and stored.")

class H5CoresetDataset(Dataset):
    """Dataset for loading a coreset defined by indices from an H5 file."""
    def __init__(self, h5_path, split, indices, saliency_weights=None):
        self.h5_path = h5_path
        self.split = split
        self.indices = indices
        self.saliency_weights = saliency_weights
        if self.saliency_weights is not None:
            assert len(self.indices) == len(self.saliency_weights)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        with h5py.File(self.h5_path, 'r') as f:
            current_pos = 0
            for scene_name in sorted(f[self.split].keys()):
                group = f[f'{self.split}/{scene_name}']
                num_points = group['coords'].shape[0]
                if current_pos + num_points > global_idx:
                    local_idx = global_idx - current_pos
                    coords = group['coords'][local_idx]
                    normals = group['normals'][local_idx]
                    label = group['labels'][local_idx]
                    features = np.concatenate([coords, normals]).astype(np.float32)
                    break
                current_pos += num_points
        
        item = (torch.from_numpy(features), torch.tensor(label, dtype=torch.long))
        if self.saliency_weights is not None:
            item += (torch.tensor(self.saliency_weights[idx], dtype=torch.float32),)
        return item

def stream_h5_data(h5_path, split, batch_size, cols=('coords', 'normals', 'labels')):
    """Generator to stream data from H5 file chunk by chunk."""
    with h5py.File(h5_path, 'r') as f:
        for scene_name in sorted(f[split].keys()):
            group = f[f'{split}/{scene_name}']
            num_points = group['coords'].shape[0]
            for i in range(0, num_points, batch_size):
                end = min(i + batch_size, num_points)
                data = {col: group[col][i:end] for col in cols}
                yield data

def random_sampler(total_points, coreset_size):
    print(f"  -> Performing Random Sampling for {coreset_size} points...")
    return np.random.choice(total_points, coreset_size, replace=False)

def grid_sampler(h5_path, coreset_size, num_train_points):
    print(f"  -> Performing Voxel Grid Sampling for {coreset_size} points...")
    ratio = coreset_size / num_train_points
    voxel_size = (1/ratio)**(1/3) * 0.05 # Heuristic
    
    all_coords = []
    for data_batch in stream_h5_data(h5_path, 'train', 100000, cols=('coords',)):
        all_coords.append(data_batch['coords'])
    all_coords = np.concatenate(all_coords, axis=0)

    _, indices = np.unique((all_coords / voxel_size).astype(int), axis=0, return_index=True)
    if len(indices) > coreset_size:
        indices = np.random.choice(indices, coreset_size, replace=False)
    return indices

def approximate_fps_sampler(h5_path, coreset_size, num_train_points):
    print(f"  -> Performing Approximate FPS for {coreset_size} points...")
    num_chunks = 20
    points_per_chunk = num_train_points // num_chunks
    coreset_per_chunk = coreset_size // num_chunks

    selected_indices = []
    offset = 0
    for data_batch in stream_h5_data(h5_path, 'train', points_per_chunk, cols=('coords',)):
        coords = data_batch['coords']
        if coords.shape[0] == 0: continue
        
        n = coords.shape[0]
        picked_indices = np.zeros(coreset_per_chunk, dtype=int)
        farthest = 0
        distances = np.full(n, np.inf)
        for i in range(coreset_per_chunk):
            picked_indices[i] = farthest
            farthest_point = coords[farthest]
            dist = np.sum((coords - farthest_point) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            farthest = np.argmax(distances)
        
        selected_indices.append(picked_indices + offset)
        offset += n
    return np.concatenate(selected_indices)

def nystrom_sampler(h5_path, coreset_size, training_batch_size, num_train_points):
    print(f"  -> Performing Nystrom-style Sampling for {coreset_size} points...")
    kmeans = MiniBatchKMeans(n_clusters=coreset_size, batch_size=training_batch_size, n_init=1, random_state=0)
    for data_batch in stream_h5_data(h5_path, 'train', training_batch_size, cols=('coords', 'normals')):
        features = np.concatenate([data_batch['coords'], data_batch['normals']], axis=1)
        kmeans.partial_fit(features)
    
    print("    (Using random sampling as a proxy for Nystrom's point selection part)")
    return random_sampler(num_train_points, coreset_size)
