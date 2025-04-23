from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNIST(Dataset):
    """
    MNIST Dataset Wrapper
    This class provides a wrapper around the MNIST dataset, allowing for easy access
    to either training or testing data. It supports indexing and provides the total length
    of the dataset.
    Attributes:
        data (torch.utils.data.Dataset): The MNIST dataset (either training or testing).
    Methods:
        __init__(data_path: str = './data', train: bool = False, flatten: bool = True):
            Initializes the MNIST dataset wrapper by downloading and preparing
            either the training or testing dataset based on the `train` flag.
        __len__():
            Returns the total number of samples in the dataset.
        __getitem__(idx: int):
            Retrieves a single sample from the dataset based on the provided index.
    Args:
        data_path (str): The path where the MNIST dataset will be downloaded and stored.
                         Defaults to './data'.
        train (bool): A flag indicating whether to load the training dataset (True)
                      or the testing dataset (False). Defaults to False.
        flatten (bool): A flag indicating whether to flatten the images into 1D tensors.
                        Defaults to True.
    Usage:
        train_dataset = MNIST(data_path='./data', train=True)
        print(len(train_dataset))  # Total number of samples in the training dataset
        sample = train_dataset[0]  # Get the first sample from the training dataset
    """
    def __init__(self, data_path: str = './data', train: bool = False, flatten: bool = True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)) if flatten else lambda x: x
        ])
        self.data = datasets.MNIST(root=data_path, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
