import torch


def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss={avg_loss:.4f}, Test Accuracy={accuracy:.2f}%")
    return avg_loss, accuracy
