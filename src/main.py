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
        self.create_dataloader()
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

    def create_dataloader(self):
        # Create data loaders.
        self.train_dataloader = DataLoader(
            self.training_data, batch_size=FLAGS.model.batch_size
        )
        self.test_dataloader = DataLoader(
            self.test_data, batch_size=FLAGS.model.batch_size
        )

        print(f"test size: {len(self.train_dataloader)}")
        for X, y in self.train_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

    def start(self):
        ...
