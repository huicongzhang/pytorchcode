import torch
import torch.optim
from torchvision import datasets,transforms
batch_size = 32
train_dataset = datasets.MNIST(
    root='./data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

