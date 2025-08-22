import torch
from torch.utils.data import Dataset

class MockNYUv2Dataset(Dataset):
    """A mock dataset that generates random data with the same shape as NYUv2."""
    def __init__(self, num_samples=160, test_mode=False, batch_size_for_test=8):
        if test_mode:
            self.num_samples = batch_size_for_test * 2
        else:
            self.num_samples = num_samples
        self.img_size = (128, 128)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, *self.img_size)
        seg_target = torch.randint(0, 13, self.img_size, dtype=torch.long)
        depth_target = torch.randn(1, *self.img_size)
        return image, {"segmentation": seg_target, "depth": depth_target}
