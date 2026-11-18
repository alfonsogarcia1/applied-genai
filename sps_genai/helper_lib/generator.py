import math

import torch

from helper_lib.data_loader import UNKNOWN
from helper_lib.model import GAN, DiffusionModel, EnergyModel


def langevin_sample(model, images, steps=60, step_size=10.0, noise_std=0.005):
    # Langevin dynamics: walk the images downhill on the energy surface.
    # This is gradient descent like in training, except the weights stay fixed and the
    # pixels are what changes. Images are in [-1, 1].
    was_training = model.training
    model.eval()
    # Freeze the weights: during sampling we only want gradients with respect to the
    # image, so autograd doesn't need to track the parameters at all
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    images = images.detach()
    try:
        # Sampling needs gradients even when the caller is inside torch.no_grad()
        with torch.enable_grad():
            for _ in range(steps):
                # A little random noise each step, so samples don't all land in the same dip
                images = (images + noise_std * torch.randn_like(images)).clamp(-1.0, 1.0)
                # The image is now the tensor whose gradient we want
                images.requires_grad_(True)
                energy = model(images)
                # d(energy)/d(image). Summing gives the gradient for every image in the batch
                # at once, because each image only affects its own energy
                grads, = torch.autograd.grad(energy.sum(), images)
                # Clip the gradient for stability, take a small step downhill, and detach so
                # the next step starts a fresh graph instead of growing this one
                images = (images - step_size * grads.clamp(-0.03, 0.03)).clamp(-1.0, 1.0).detach()
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        model.train(was_training)
    return images


def generate_ebm_images(model, num_images, steps=256, step_size=10.0, noise_std=0.01, device="cpu"):
    # Start from uniform noise in [-1, 1] and run a long Langevin chain, then map to [0, 1]
    start = torch.rand(num_images, model.num_channels, model.image_size, model.image_size, device=device) * 2 - 1
    images = langevin_sample(model, start, steps, step_size, noise_std)
    return (images + 1) / 2


def generate_samples(model, device, num_samples=10, diffusion_steps=100):
    # Imported here so the API (which has no matplotlib) can still use generate_text
    import matplotlib.pyplot as plt

    model.to(device)
    model.eval()
    if isinstance(model, EnergyModel):
        # Not inside no_grad: Langevin sampling needs the gradient with respect to the image
        samples = generate_ebm_images(model, num_samples, device=device)
    else:
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
