import os
from torchvision.datasets import MNIST
data_dir = 'Data'

os.makedirs(data_dir, exist_ok = True)

MNIST(root=data_dir, train=True, download=True)
MNIST(root=data_dir, train=False, download=True)

