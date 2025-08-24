import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

dataset = torchvision.datasets.ImageFolder(root="/path/to/images", transform=transform)