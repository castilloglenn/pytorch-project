import torch
from absl import flags
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

FLAGS = flags.FLAGS


class Main:
    def __init__(self) -> None:
        self.download_fashion_mnist()
        self.start()

    def download_fashion_mnist(self):
        # Download training data from open datasets.
        self.training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        self.test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

    def start(self):
        print(self.training_data)
        print(self.test_data)
