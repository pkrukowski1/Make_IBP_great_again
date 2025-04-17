from torch.utils.data import Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

class CIFAR10(Dataset):
    """
    CIFAR10 Dataset Wrapper
    This class provides a wrapper around the CIFAR-10 dataset, allowing for easy access
    to either training or testing data. It supports indexing and provides the total length
    of the dataset.
    Attributes:
        data (torch.utils.data.Dataset): The CIFAR-10 dataset (either training or testing).
    Methods:
        __init__(data_path: str = './data', train: bool = False):
            Initializes the CIFAR10 dataset wrapper by downloading and preparing
            either the training or testing dataset based on the `train` flag.
        __len__():
            Returns the total number of samples in the dataset.
        __getitem__(idx: int):
            Retrieves a single sample from the dataset based on the provided index.
    Args:
        data_path (str): The path where the CIFAR-10 dataset will be downloaded and stored.
                         Defaults to './data'.
        train (bool): A flag indicating whether to load the training dataset (True)
                      or the testing dataset (False). Defaults to False.
    Usage:
        train_dataset = CIFAR10(data_path='./data', train=True)
        print(len(train_dataset))  # Total number of samples in the training dataset
        sample = train_dataset[0]  # Get the first sample from the training dataset
    """

    def __init__(self, data_path: str = './data', train: bool = False):
        self.data = datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]