import torch
import torch.utils.data

def create_mock_dataloader(num_samples, batch_size):
    """Creates a dataloader with random data and a 'difficulty' metric."""
    features = torch.randn(num_samples, 10)
    labels = torch.randint(0, 10, (num_samples,))
    # Simulate some inputs being harder than others
    difficulty = torch.randint(0, 2, (num_samples,)).float() # 0 for easy, 1 for hard
    dataset = torch.utils.data.TensorDataset(features, labels, difficulty)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader
