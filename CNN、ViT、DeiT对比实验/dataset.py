import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


def get_cifar10_loaders(batch_size=128):

    # ======================
    # Train Transform
    # ======================
    train_transform = transforms.Compose([

        transforms.RandomCrop(32, padding=4),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    # ======================
    # Test Transform
    # ======================
    test_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    # ======================
    # Dataset
    # ======================
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=test_transform
    )

    # ======================
    # Loader
    # ======================
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader