import torch
import torchvision

# Load data and transform 

from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms


## Define a transform to read the data in as a tensor
data_transform = transforms.ToTensor()
# data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomRotation(30,resample=False, expand=False, center=None),transforms.ToTensor()])

# choose the training and test datasets
train_data = FashionMNIST(root='./data', train=True,
                                download=True, transform=data_transform)

test_data = FashionMNIST(root='./data', train=False,
                                 download=True, transform=data_transform)
