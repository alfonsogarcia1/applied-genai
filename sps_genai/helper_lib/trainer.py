import torch
import torch.nn.functional as F
from tqdm import tqdm

from helper_lib.generator import langevin_sample


def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch + 1}: Loss={running_loss / len(data_loader):.4f}, "
              f"Accuracy={100 * correct / total:.2f}%")
    return model


def vae_loss_function(recon_x, x, mu, logvar, beta=500):
    # Reconstruction error plus KL divergence from a standard normal.
    # beta weights the reconstruction term, same as the class notebook.
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * bce + kld


def train_vae_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # Same loop as train_model, but the target is the input image itself
    # and criterion takes (recon, x, mu, logvar), e.g. vae_loss_function
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for inputs, _ in progress:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss = criterion(recon, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch + 1}: Loss={running_loss / len(data_loader):.4f}")
    return model


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # model is a GAN, criterion is BCEWithLogitsLoss, and optimizer is a dict with one
    # optimizer per network: {"generator": ..., "discriminator": ...}
    model.to(device)
    model.train()
    generator, discriminator = model.generator, model.discriminator
    for epoch in range(epochs):
        d_running, g_running = 0.0, 0.0
        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for real, _ in progress:
            real = real.to(device)
            batch_size = real.size(0)
            ones = torch.ones(batch_size, 1, device=device)
            zeros = torch.zeros(batch_size, 1, device=device)

            # Discriminator: label real images 1 and generated images 0.
            # detach() so this step does not update the generator
            z = torch.randn(batch_size, model.z_dim, device=device)
            fake = generator(z)
            d_loss = criterion(discriminator(real), ones) + criterion(discriminator(fake.detach()), zeros)
            optimizer["discriminator"].zero_grad()
            d_loss.backward()
            optimizer["discriminator"].step()

            # Generator: wants the discriminator to label its images 1
            g_loss = criterion(discriminator(fake), ones)
            optimizer["generator"].zero_grad()
            g_loss.backward()
            optimizer["generator"].step()

            d_running += d_loss.item()
            g_running += g_loss.item()
            progress.set_postfix(d_loss=f"{d_loss.item():.4f}", g_loss=f"{g_loss.item():.4f}")

        print(f"Epoch {epoch + 1}: D loss={d_running / len(data_loader):.4f}, "
              f"G loss={g_running / len(data_loader):.4f}")
    return model


def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10,
                    max_grad_norm=1.0, checkpoint_path=None):
    # model is a DiffusionModel, optimizer holds model.network.parameters().
    # Add a random amount of noise to each image and train the UNet to predict that noise.
    # checkpoint_path, if given, gets the weights after every epoch
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, _ in progress:
            images = model.normalize(images.to(device))
            noises = torch.randn_like(images)
            diffusion_times = torch.rand(images.size(0), 1, 1, 1, device=device)
            noise_rates, signal_rates = model.schedule(diffusion_times)
            noisy_images = signal_rates * images + noise_rates * noises

            optimizer.zero_grad()
            pred_noises, _ = model.denoise(noisy_images, noise_rates, signal_rates, training=True)
            loss = criterion(pred_noises, noises)
            # Stop straight away if training blows up, instead of saving NaN weights
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss became {loss.item()} in epoch {epoch + 1}")
            loss.backward()
            # The UNet has no normalization layers, so cap the gradient size to keep one
            # bad batch from throwing the weights off
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), max_grad_norm)
            optimizer.step()
            model.update_ema()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch + 1}: Loss={running_loss / len(data_loader):.4f}", flush=True)
        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path)
    return model


def ebm_loss_function(real_energy, fake_energy, alpha=0.1):
    # Contrastive divergence: push the energy of real images down and of sampled images up.
    # The partition function Z cancels out, so it never has to be computed.
    # The alpha term keeps the energies near 0 so they can't drift off to +/- infinity
    cdiv_loss = real_energy.mean() - fake_energy.mean()
    reg_loss = alpha * (real_energy.pow(2).mean() + fake_energy.pow(2).mean())
    return cdiv_loss + reg_loss


class ReplayBuffer:
    # Langevin chains need many steps to reach good images. Instead of starting from noise
    # every batch, keep recent samples and continue from them (plus ~5% fresh noise)
    def __init__(self, image_shape, max_size=8192, device='cpu'):
        self.image_shape = image_shape
        self.max_size = max_size
        self.device = device
        self.examples = torch.rand(128, *image_shape, device=device) * 2 - 1

    def sample(self, model, batch_size, steps, step_size, noise_std):
        n_new = int(torch.binomial(torch.tensor(float(batch_size)), torch.tensor(0.05)))
        fresh = torch.rand(n_new, *self.image_shape, device=self.device) * 2 - 1
        old = self.examples[torch.randint(len(self.examples), (batch_size - n_new,), device=self.device)]
        new_images = langevin_sample(model, torch.cat([fresh, old]), steps, step_size, noise_std)
        # Newest samples go first, the oldest fall off the end
        self.examples = torch.cat([new_images, self.examples])[:self.max_size]
        return new_images


def train_ebm(model, data_loader, criterion, optimizer, device='cpu', epochs=10,
              steps=60, step_size=10.0, noise_std=0.005):
    # model is an EnergyModel, criterion is ebm_loss_function, images are in [-1, 1]
    model.to(device)
    buffer = ReplayBuffer((model.num_channels, model.image_size, model.image_size), device=device)
    for epoch in range(epochs):
        running_loss, running_real, running_fake = 0.0, 0.0, 0.0
        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for real, _ in progress:
            real = real.to(device)
            # A little noise on the real images too, so they aren't perfectly sharp targets
            real = (real + noise_std * torch.randn_like(real)).clamp(-1.0, 1.0)
            # Fake images come from Langevin sampling, which changes the images, not the weights
            fake = buffer.sample(model, real.size(0), steps, step_size, noise_std)

            # Now the usual training step: gradients with respect to the weights only.
            # fake is already detached, so no gradient flows back into the sampler
            model.train()
            energy = model(torch.cat([real, fake]))
            real_energy, fake_energy = energy[:real.size(0)], energy[real.size(0):]
            loss = criterion(real_energy, fake_energy)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            running_loss += loss.item()
            running_real += real_energy.mean().item()
            running_fake += fake_energy.mean().item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        n = len(data_loader)
        print(f"Epoch {epoch + 1}: Loss={running_loss / n:.4f}, "
              f"real energy={running_real / n:.4f}, fake energy={running_fake / n:.4f}")
    return model


def train_rnn_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # Next word prediction: every position in the sequence is scored against the word after it
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            # Flatten (batch, seq_len, vocab) to (batch * seq_len, vocab) for CrossEntropyLoss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch + 1}: Loss={running_loss / len(data_loader):.4f}")
    return model
