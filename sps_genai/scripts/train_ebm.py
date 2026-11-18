# Assignment 4: train the energy-based model on CIFAR10 and save the weights the API loads
# Run from sps_genai/:  uv run python -m scripts.train_ebm
import torch
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import ebm_loss_function, train_ebm
from helper_lib.utils import get_device, save_model

EPOCHS = 20
WEIGHTS_PATH = "app/weights/ebm_cifar10.pth"

torch.manual_seed(42)
device = get_device()

# The sampler works in [-1, 1], so the real images have to be in the same range
train_loader = get_data_loader('data', batch_size=128, dataset="CIFAR10", minus_one_to_one=True)

model = get_model("EBM", num_channels=3, image_size=32)
# Settings from the class notebook. beta1 = 0 turns off momentum, because the fake images
# change every step and old gradients quickly stop pointing the right way
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))

train_ebm(model, train_loader, ebm_loss_function, optimizer, device=device, epochs=EPOCHS,
          steps=60, step_size=10.0, noise_std=0.005)

save_model(model.to("cpu"), WEIGHTS_PATH)
print(f"Saved weights to {WEIGHTS_PATH}")
