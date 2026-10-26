# Assignment 3: train the GAN on MNIST and save the weights the API loads
# Run from sps_genai/:  uv run python -m scripts.train_gan
import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import get_device, save_model

EPOCHS = 20
WEIGHTS_PATH = "app/weights/gan_mnist.pth"

torch.manual_seed(42)
device = get_device()

train_loader = get_data_loader('data', batch_size=128, dataset="MNIST")

gan = get_model("GAN")
criterion = nn.BCEWithLogitsLoss()
# lr 0.0002 and beta1 0.5 are the usual Adam settings for this kind of GAN (DCGAN paper)
optimizer = {
    "generator": optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
    "discriminator": optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
}

trained_gan = train_gan(gan, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)

# Save on the CPU so the weights load anywhere, including the Docker container
save_model(trained_gan.to("cpu"), WEIGHTS_PATH)
print(f"Saved weights to {WEIGHTS_PATH}")
