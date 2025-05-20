import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size: int, num_workers: int = 4):
    """
    MNIST 용 DataLoader를 반환합니다.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    val_ds   = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

