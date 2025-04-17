from torch.utils.data import Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

class MNIST(Dataset):
    """
    A dataset class for handling MNIST data.

    This class provides an interface to load and interact with the MNIST dataset. It supports both training and testing modes 
    and allows easy access to individual samples.

        data (Dataset): The MNIST dataset loaded from torchvision, either in training or testing mode.

            Returns the total number of samples in the dataset.
        __getitem__(idx: int):
            Retrieves a sample (image and label) from the dataset based on the given index.

        train (bool): A flag indicating whether to load the training dataset (True) or the testing dataset (False). Defaults to True.
    """
    def __init__(self, data_path: str = './data', train: bool = False):
        self.data = datasets.MNIST(root=data_path, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
