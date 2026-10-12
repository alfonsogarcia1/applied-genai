import os

import torch


def get_device():
    # Apple GPU, then NVIDIA GPU, then CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_model(model, path="models/model.pth"):
    # Saves only the weights, rebuild the model with get_model before loading
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir="checkpoints"):
    # Weights plus optimizer state, so training can resume later
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
    torch.save(checkpoint, path)
    return path


def load_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
