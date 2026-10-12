import torch
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True, dataset="CIFAR10", image_size=None):
    if dataset == "CIFAR10":
        # 3 x 32 x 32 color images, 10 classes. image_size resizes them, e.g. 64 for CNN64
        steps = [transforms.Resize((image_size, image_size))] if image_size else []
        transform = transforms.Compose(steps + [transforms.ToTensor()])
        data = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    elif dataset == "FashionMNIST":
        # Images are 28 x 28, pad to 32 x 32 so the VAE encoder halves cleanly to 4 x 4
        transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
        data = datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Only shuffle the training data
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=train)
    return loader
