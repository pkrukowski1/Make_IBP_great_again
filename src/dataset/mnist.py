from torch.utils.data import Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

class MNIST(Dataset):
    class MNIST:
        """
        A dataset class for handling MNIST-like data using CIFAR-10 as a placeholder.
        Attributes:
            train_data (Dataset): The training dataset loaded from CIFAR-10.
            test_data (Dataset): The testing dataset loaded from CIFAR-10.
        Methods:
            __len__():
                Returns the total number of samples in both training and testing datasets.
            __getitem__(idx: int, train: bool):
                Retrieves a sample from the training or testing dataset based on the index and mode.
        Args:
            data_path (str): The path where the dataset will be downloaded or loaded from. Defaults to './data'.
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
