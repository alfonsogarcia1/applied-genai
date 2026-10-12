# Module 5 activity: train a VAE on Fashion-MNIST and generate new images
# Run from sps_genai/:  uv run python -m scripts.train_vae
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_vae_model, vae_loss_function
from helper_lib.utils import get_device, save_model

device = get_device()

train_loader = get_data_loader('data', batch_size=64, dataset="FashionMNIST")

vae = get_model("VAE")
criterion = vae_loss_function
optimizer = optim.Adam(vae.parameters(), lr=0.001)

train_vae_model(vae, train_loader, criterion, optimizer, device=device, epochs=5)
save_model(vae, "models/vae.pth")
generate_samples(vae, device, num_samples=10)
