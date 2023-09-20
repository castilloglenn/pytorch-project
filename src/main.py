import torch
from absl import flags
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

FLAGS = flags.FLAGS

# Get cpu, gpu or mps device for training.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Main:
    def __init__(self) -> None:
        self.download_fashion_mnist()
        self.create_dataloader()
        self.create_model_instance()
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

        print(f"Dataset type: {type(self.test_data)}")
        print()

    def create_dataloader(self):
        # Create data loaders.
        self.train_dataloader = DataLoader(
            self.training_data, batch_size=FLAGS.model.batch_size
        )
        self.test_dataloader = DataLoader(
            self.test_data, batch_size=FLAGS.model.batch_size
        )

        print(f"test size: {len(self.test_dataloader)}")
        for X, y in self.test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        print(f"Dataloader type: {type(self.test_dataloader)}")
        print()

    def create_model_instance(self):
        self.model = NeuralNetwork().to(DEVICE)
        print(self.model)
        print()

    def start(self):
        ...
