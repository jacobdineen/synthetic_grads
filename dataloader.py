import torchvision
import torch.utils.data as data_utils
import torch


def get_dataloaders(
    train_size: int = 5000, test_size: int = 1000, batch_size_train: int = 256,
):
    indices = torch.arange(train_size)
    train = torchvision.datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize(48), torchvision.transforms.ToTensor(),]
        ),
    )

    train_slice = data_utils.Subset(train, indices)

    train_loader = torch.utils.data.DataLoader(
        train_slice, batch_size=batch_size_train, shuffle=True
    )

    indices = torch.arange(test_size)
    test = torchvision.datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(48),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    test_slice = data_utils.Subset(test, indices)

    test_loader = torch.utils.data.DataLoader(
        test_slice, batch_size=test_size, shuffle=True
    )
    return train_loader, test_loader
