# Module 8 activity: train a diffusion model on Oxford Flowers102 and generate new images
# Run from sps_genai/:  uv run python -m scripts.train_diffusion
# The first run downloads Flowers102 (~350 MB) into data/
import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.utils import get_device, save_model

EPOCHS = 50

torch.manual_seed(42)
device = get_device()

train_loader = get_data_loader('data', batch_size=64, dataset="Flowers102")

model = get_model("Diffusion", image_size=64, num_channels=3)

# Per-channel mean and std of the training images, so the model works on mean 0, variance 1
images = torch.cat([batch for batch, _ in train_loader])
model.set_normalizer(images.mean(dim=(0, 2, 3)), images.std(dim=(0, 2, 3)))

optimizer = optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4)
train_diffusion(model, train_loader, nn.L1Loss(), optimizer, device=device, epochs=EPOCHS)

save_model(model.to("cpu"), "models/diffusion.pth")
generate_samples(model, "cpu", num_samples=10, diffusion_steps=100)
