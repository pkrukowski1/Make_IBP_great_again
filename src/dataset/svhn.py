from torch.utils.data import Dataset
from torchvision import datasets, transforms


class SVHN(Dataset):
    """
    SVHN Dataset class for loading and accessing the Street View House Numbers (SVHN) dataset.
    Attributes:
        data (torch.utils.data.Dataset): The dataset loaded from the SVHN dataset, either training or testing.
    Methods:
        __init__(data_path: str = './data', train: bool = False, flatten: bool = True):
            Initializes the SVHN dataset by downloading and preparing the specified split (train or test).
        __len__() -> int:
            Returns the total number of samples in the dataset.
        __getitem__(idx: int):
            Retrieves a sample from the dataset at the specified index.
    Args:
        data_path (str): The root directory where the SVHN dataset will be downloaded and stored. Defaults to './data'.
        train (bool): Whether to load the training split. If False, loads the testing split. Defaults to False.
    """

    def __init__(self, data_path: str= './data', train: bool = False, flatten: bool = True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)) if flatten else lambda x: x
        ])
        _split = "train" if train else "test"
        self.data = datasets.SVHN(root=data_path, split=_split, download=True, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]