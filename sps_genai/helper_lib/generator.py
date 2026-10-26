import math

import torch

from helper_lib.data_loader import UNKNOWN
from helper_lib.model import GAN, DiffusionModel


def generate_samples(model, device, num_samples=10, diffusion_steps=100):
    # Imported here so the API (which has no matplotlib) can still use generate_text
    import matplotlib.pyplot as plt

    model.to(device)
    model.eval()
    with torch.no_grad():
        if isinstance(model, GAN):
            # Random noise vectors through the generator, then [-1, 1] back to [0, 1]
            z = torch.randn(num_samples, model.z_dim).to(device)
            samples = (model.generator(z) + 1) / 2
        elif isinstance(model, DiffusionModel):
            # Start from standard normal noise and run the reverse diffusion
            samples = model.generate(num_samples, diffusion_steps)
        else:
            # VAE: sample points from a standard normal in the latent space and decode them
            z = torch.randn(num_samples, model.latent_dim).to(device)
            samples = model.decoder(z)
    # (channels, height, width) -> (height, width, channels) for imshow
    samples = samples.permute(0, 2, 3, 1).cpu().numpy()

    cols = min(num_samples, 5)
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(samples[i].squeeze(), cmap='gray')  # cmap only applies to 1-channel images
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def generate_text(model, vocab, start_word, length=20, temperature=1.0, device='cpu'):
    # Feed the start word in, then keep feeding back the sampled word until we have length words
    stoi = {word: i for i, word in enumerate(vocab)}
    model.to(device)
    model.eval()
    words = [start_word]
    input_ids = torch.tensor([[stoi[start_word]]], device=device)
    hidden = None

    with torch.no_grad():
        for _ in range(length - 1):
            output, hidden = model(input_ids, hidden)
            logits = output[:, -1, :] / temperature
            # Never pick the unknown word placeholder
            logits[:, stoi[UNKNOWN]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            words.append(vocab[next_id])
            input_ids = torch.tensor([[next_id]], device=device)

    return " ".join(words)
