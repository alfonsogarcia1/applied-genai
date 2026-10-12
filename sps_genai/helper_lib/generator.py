import math

import matplotlib.pyplot as plt
import torch


def generate_samples(model, device, num_samples=10):
    # Sample points from a standard normal in the latent space and decode them
    model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decoder(z).cpu().numpy()

    cols = min(num_samples, 5)
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
