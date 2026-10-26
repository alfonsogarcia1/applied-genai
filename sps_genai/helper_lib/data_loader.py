import re
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

UNKNOWN = "<unk>"


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
    elif dataset == "MNIST":
        # 1 x 28 x 28 digits scaled to [-1, 1] to match the GAN generator's Tanh output
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        data = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    elif dataset == "Flowers102":
        # Photos of different sizes: crop the middle square and shrink it to 64 x 64
        transform = transforms.Compose([
            transforms.CenterCrop(340), transforms.Resize((64, 64)), transforms.ToTensor(),
        ])
        split = "train" if train else "val"
        data = datasets.Flowers102(root=data_dir, split=split, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Only shuffle the training data
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=train)
    return loader


def tokenize(text):
    # Lowercase words only, same as the bigram model
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(words, max_size=10000):
    # The most frequent words, most frequent first. Index 0 stands for every other word
    most_common = [word for word, _ in Counter(words).most_common(max_size - 1)]
    return [UNKNOWN] + most_common


class TextDataset(Dataset):
    # Each example is seq_len word ids, and the target is the same words shifted by one
    def __init__(self, ids, seq_len=30):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx):
        x = self.ids[idx:idx + self.seq_len]
        y = self.ids[idx + 1:idx + self.seq_len + 1]
        return x, y


def get_text_data_loader(text, vocab, batch_size=64, seq_len=30):
    stoi = {word: i for i, word in enumerate(vocab)}
    ids = [stoi.get(word, 0) for word in tokenize(text)]
    return torch.utils.data.DataLoader(TextDataset(ids, seq_len), batch_size=batch_size, shuffle=True)
