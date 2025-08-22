import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    import MinkowskiEngine as ME
    MINKOWSKI_AVAILABLE = True
except ImportError:
    print("Warning: MinkowskiEngine not found. LGS model and SICoD will not be available.")
    MINKOWSKI_AVAILABLE = False

# --- Models ---
class MockPTv3(nn.Module):
    """A simplified mock of Point Transformer V3 for demonstration."""
    def __init__(self, in_features, out_classes, model_dim, depth):
        super().__init__()
        layers = [nn.Linear(in_features, model_dim), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(model_dim, model_dim), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(model_dim, out_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

if MINKOWSKI_AVAILABLE:
    class LGSSparseUNet(ME.MinkowskiNetwork):
        """Sparse 3D U-Net for LGS saliency prediction using MinkowskiEngine."""
        def __init__(self, in_channels, out_channels, channels, D=3):
            super().__init__(D)
            self.conv1 = nn.Sequential(
                ME.MinkowskiConvolution(in_channels, channels[0], kernel_size=3, stride=1, dimension=D),
                ME.MinkowskiBatchNorm(channels[0]),
                ME.MinkowskiReLU(inplace=True)
            )
            self.down1 = nn.Sequential(
                ME.MinkowskiConvolution(channels[0], channels[1], kernel_size=3, stride=2, dimension=D),
                ME.MinkowskiBatchNorm(channels[1]),
                ME.MinkowskiReLU(inplace=True)
            )
            self.down2 = nn.Sequential(
                ME.MinkowskiConvolution(channels[1], channels[2], kernel_size=3, stride=2, dimension=D),
                ME.MinkowskiBatchNorm(channels[2]),
                ME.MinkowskiReLU(inplace=True)
            )
            self.up1 = nn.Sequential(
                ME.MinkowskiGenerativeConvolutionTranspose(channels[2], channels[1], kernel_size=3, stride=2, dimension=D),
                ME.MinkowskiBatchNorm(channels[1]),
                ME.MinkowskiReLU(inplace=True)
            )
            self.up2 = nn.Sequential(
                ME.MinkowskiGenerativeConvolutionTranspose(channels[1], channels[0], kernel_size=3, stride=2, dimension=D),
                ME.MinkowskiBatchNorm(channels[0]),
                ME.MinkowskiReLU(inplace=True)
            )
            self.final_conv = ME.MinkowskiConvolution(channels[0], out_channels, kernel_size=1, dimension=D)

        def forward(self, x):
            out1 = self.conv1(x)
            out_down1 = self.down1(out1)
            out_down2 = self.down2(out_down1)
            out_up1 = self.up1(out_down2)
            out_up2 = self.up2(out_up1 + out_down1)
            out = self.final_conv(out_up2 + out1)
            return out

# --- Loss Function ---
class WeightedCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets, weights):
        unweighted_loss = self.base_loss(logits, targets)
        weighted_loss = unweighted_loss * weights
        return weighted_loss.mean()

# --- Training & Evaluation ---
def train_teacher(model, train_loader, epochs, lr, device, weighted=False):
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = WeightedCELoss() if weighted else nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            if weighted:
                features, labels, weights = batch
                features, labels, weights = features.to(device), labels.to(device), weights.to(device)
            else:
                features, labels = batch
                features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            if weighted:
                loss = criterion(logits, labels, weights)
            else:
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(f'    Teacher Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        scheduler.step()
    return history

if MINKOWSKI_AVAILABLE:
    def train_lgs(model, h5_path, saliency_targets, stream_h5_data_func, config, device):
        model.to(device)
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=config['training']['lgs_lr'])
        criterion = nn.MSELoss()
        voxel_size = config['model']['lgs_voxel_size']
        batch_size = config['training']['batch_size']
        epochs = config['training']['lgs_epochs']
        history = []
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            offset = 0
            for data_batch in stream_h5_data_func(h5_path, 'train', batch_size, cols=('coords', 'normals')):
                coords = data_batch['coords']
                local_coords = coords - coords.mean(axis=0)
                geom_features = np.concatenate([local_coords, data_batch['normals']], axis=1)
                
                targets = saliency_targets[offset : offset + len(coords)]
                offset += len(coords)
                if offset >= len(saliency_targets): 
                    # This logic assumes the entire dataset is streamed once per epoch
                    offset = 0 

                mink_coords = ME.utils.batched_coordinates([coords / voxel_size])
                features_tensor = torch.from_numpy(geom_features).float().to(device)
                targets_tensor = torch.from_numpy(targets).float().to(device)

                sparse_input = ME.SparseTensor(features_tensor, coordinates=mink_coords, device=device)
                
                optimizer.zero_grad()
                pred_saliency = model(sparse_input).features.squeeze()
                
                if pred_saliency.shape[0] != targets_tensor.shape[0]: continue
                
                loss = criterion(pred_saliency, targets_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                history.append(avg_loss)
                print(f'    LGS Epoch {epoch+1}/{epochs}, MSE Loss: {avg_loss:.6f}')
        return history
