import os
import warnings
from torch.utils.data import DataLoader

# Suppress torchmeta warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchmeta')

try:
    from torchmeta.datasets.helpers import miniimagenet, tieredimagenet
    from torchmeta.utils.data import BatchMetaDataLoader
except ImportError:
    print("torchmeta not found. Please install it with: pip install torchmeta")
    exit()

def get_dataset(config):
    """
    Loads the specified dataset using torchmeta.
    """
    dataset_name = config['experiment']['dataset']
    data_path = config['experiment']['data_path']
    n_way = config['experiment']['n_way']
    k_shot = config['experiment']['k_shot']
    batch_size = config['training']['batch_size']
    num_workers = config['system']['num_workers']

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if dataset_name == 'miniimagenet':
        dataset_fn = miniimagenet
    elif dataset_name == 'tieredimagenet':
        dataset_fn = tieredimagenet
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    # The number of test shots (query examples) per class in each task
    # Standard is 15 for mini/tieredimagenet
    test_shots = 15 

    train_dataset = dataset_fn(data_path, ways=n_way, shots=k_shot, test_shots=test_shots, meta_split='train', download=True)
    val_dataset = dataset_fn(data_path, ways=n_way, shots=k_shot, test_shots=test_shots, meta_split='val', download=True)
    test_dataset = dataset_fn(data_path, ways=n_way, shots=k_shot, test_shots=test_shots, meta_split='test', download=True)

    train_loader = BatchMetaDataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = BatchMetaDataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    # Use batch_size=1 for testing to evaluate one task at a time
    test_loader = BatchMetaDataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False)
    
    print(f"Loaded {dataset_name.capitalize()} dataset.")
    print(f"Train tasks: {len(train_dataset)}, Val tasks: {len(val_dataset)}, Test tasks: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
