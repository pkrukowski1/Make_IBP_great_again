from torch.utils.data import Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

class SVHN(Dataset):
    """
    SVHN Dataset class for loading and accessing the Street View House Numbers (SVHN) dataset.
    Attributes:
        train_data (torch.utils.data.Dataset): The training dataset loaded from the SVHN dataset.
        test_data (torch.utils.data.Dataset): The testing dataset loaded from the SVHN dataset.
    Methods:
        __init__(data_path: str = './data'):
            Initializes the SVHN dataset by downloading and preparing the training and testing datasets.
        __len__() -> int:
            Returns the total number of samples in both the training and testing datasets.
        __getitem__(idx: int, train: bool):
            Retrieves a sample from the dataset. If `train` is True, retrieves a sample from the training dataset.
            Otherwise, retrieves a sample from the testing dataset.
    Args:
        data_path (str): The root directory where the SVHN dataset will be downloaded and stored. Defaults to './data'.
    """

    def __init__(self, data_path: str= './data'):
        self.train_data = datasets.SVHN(root=data_path, split="train", download=True, transform=transform)
        self.test_data = datasets.SVHN(root=data_path, split="test", download=True, transform=transform)

    def __len__(self):
        return len(self.train_data) + len(self.test_data)

    def __getitem__(self, idx: int, train: bool):
        if train:
            idx = idx % len(self.train_data)
            return self.train_data[idx]
        else:
            idx = idx % len(self.test_data)
            return self.test_data[idx]