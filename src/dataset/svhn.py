from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class SVHN(Dataset):
    """
    SVHN Dataset class for loading and accessing the Street View House Numbers (SVHN) dataset.
    Attributes:
        data (torch.utils.data.Dataset): The dataset loaded from the SVHN dataset, either training or testing.
        no_points (int or None): Optional limit on the number of data points used.
    Methods:
        __init__(data_path: str = './data', train: bool = False, flatten: bool = True, no_points=None):
            Initializes the SVHN dataset by downloading and preparing the specified split (train or test).
        __len__() -> int:
            Returns the total number of samples in the dataset.
        __getitem__(idx: int):
            Retrieves a sample from the dataset at the specified index.
    """

    def __init__(self, data_path: str = './data', train: bool = False, flatten: bool = True, no_points=None):
        if isinstance(no_points, (list, tuple)):
            no_points = no_points[0]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)) if flatten else lambda x: x
        ])

        _split = "train" if train else "test"
        full_data = datasets.SVHN(root=data_path, split=_split, download=True, transform=transform)

        self.data = full_data if no_points is None else Subset(full_data, range(no_points))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
