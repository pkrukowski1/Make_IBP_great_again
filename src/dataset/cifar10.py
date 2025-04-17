from torch.utils.data import Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

class CIFAR10(Dataset):
    """
    CIFAR10 Dataset Wrapper
    This class provides a wrapper around the CIFAR-10 dataset, allowing for easy access
    to both training and testing data. It supports indexing and provides the total length
    of the combined dataset.
    Attributes:
        train_data (torch.utils.data.Dataset): The training dataset for CIFAR-10.
        test_data (torch.utils.data.Dataset): The testing dataset for CIFAR-10.
    Methods:
        __init__(data_path: str = './data'):
            Initializes the CIFAR10 dataset wrapper by downloading and preparing
            the training and testing datasets.
        __len__():
            Returns the total number of samples in both the training and testing datasets.
        __getitem__(idx: int, train: bool):
            Retrieves a single sample from the dataset. If `train` is True, the sample
            is retrieved from the training dataset; otherwise, it is retrieved from the
            testing dataset. The index wraps around if it exceeds the dataset size.
    Args:
        data_path (str): The path where the CIFAR-10 dataset will be downloaded and stored.
                         Defaults to './data'.
    Usage:
        dataset = CIFAR10(data_path='./data')
        print(len(dataset))  # Total number of samples in training and testing datasets
        sample = dataset[0, train=True]  # Get the first sample from the training dataset
    """

    def __init__(self, data_path: str = './data'):
        self.train_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        self.test_data = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    def __len__(self):
        return len(self.train_data) + len(self.test_data)

    def __getitem__(self, idx: int, train: bool):
        if train:
            idx = idx % len(self.train_data)
            return self.train_data[idx]
        else:
            idx = idx % len(self.test_data)
            return self.test_data[idx]