# Assignment 2: train the CNN64 classifier on CIFAR10 and save the weights the API loads
# Run from sps_genai/:  uv run python -m scripts.train_classifier
import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.utils import get_device, save_model

EPOCHS = 10
WEIGHTS_PATH = "app/weights/cnn64_cifar10.pth"

torch.manual_seed(42)
device = get_device()

# CIFAR10 images are 32 x 32, the architecture expects 64 x 64
train_loader = get_data_loader('data', batch_size=64, image_size=64)
test_loader = get_data_loader('data', batch_size=64, train=False, image_size=64)

model = get_model("CNN64")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trained_model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)
evaluate_model(trained_model, test_loader, criterion, device=device)

# Save on the CPU so the weights load anywhere, including the Docker container
save_model(trained_model.to("cpu"), WEIGHTS_PATH)
print(f"Saved weights to {WEIGHTS_PATH}")
