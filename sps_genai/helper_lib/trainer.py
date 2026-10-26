import torch
import torch.nn.functional as F
from tqdm import tqdm


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


def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # model is a DiffusionModel, optimizer holds model.network.parameters().
    # Add a random amount of noise to each image and train the UNet to predict that noise
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
            loss.backward()
            optimizer.step()
            model.update_ema()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch + 1}: Loss={running_loss / len(data_loader):.4f}")
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
