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
