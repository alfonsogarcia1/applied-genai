# Assignment 4: train the diffusion model on CIFAR10 and save the weights the API loads
# Run from sps_genai/:  uv run python -m scripts.train_diffusion
import os

import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.utils import get_device, save_model

EPOCHS = 30
WEIGHTS_PATH = "app/weights/diffusion_cifar10.pth"
# Saved after every epoch, so a crash part way through doesn't lose everything
CHECKPOINT_PATH = "models/diffusion_cifar10_last.pth"

torch.manual_seed(42)
device = get_device()

# CIFAR10 pixels in [0, 1]; the model normalizes them itself
train_loader = get_data_loader('data', batch_size=64, dataset="CIFAR10")

# 0.999 averages the weights over roughly the last 1000 steps, which gives smoother images
# than the notebook's 0.8 (that one is almost the same as the live network)
model = get_model("Diffusion", image_size=32, num_channels=3, ema_decay=0.999)

# Per-channel mean and std of the training images, so the model works on mean 0, variance 1
total, total_sq, count = torch.zeros(3), torch.zeros(3), 0
for images, _ in train_loader:
    total += images.sum(dim=(0, 2, 3))
    total_sq += images.pow(2).sum(dim=(0, 2, 3))
    count += images.numel() // 3
mean = total / count
std = (total_sq / count - mean ** 2).sqrt()
print(f"Channel mean {mean.tolist()}, std {std.tolist()}")
model.set_normalizer(mean, std)

# lr 3e-4 rather than the notebook's 1e-3: at 1e-3 a first run on CIFAR10 ended with NaN weights
optimizer = optim.AdamW(model.network.parameters(), lr=3e-4, weight_decay=1e-4)
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
train_diffusion(model, train_loader, nn.L1Loss(), optimizer, device=device, epochs=EPOCHS,
                max_grad_norm=1.0, checkpoint_path=CHECKPOINT_PATH)

# Save on the CPU so the weights load anywhere, including the Docker container
save_model(model.to("cpu"), WEIGHTS_PATH)
print(f"Saved weights to {WEIGHTS_PATH}")
