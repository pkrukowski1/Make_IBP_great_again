from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import math

class MNIST(Dataset):
    def __init__(self, data_path: str = './data', train: bool = False, flatten: bool = True, split_ratio=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)) if flatten else lambda x: x
        ])

        full_dataset = datasets.MNIST(root=data_path, train=train, download=True, transform=transform)

        if isinstance(split_ratio, (list, tuple)) and len(split_ratio) == 2:
            total = len(full_dataset)
            train_size = math.floor(split_ratio[0] * total)
            indices = list(range(total))
            self.data = Subset(full_dataset, indices[:train_size])
        else:
            self.data = full_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
