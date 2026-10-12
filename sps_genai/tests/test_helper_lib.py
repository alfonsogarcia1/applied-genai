import matplotlib
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from helper_lib.data_loader import get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_model, train_vae_model, vae_loss_function
from helper_lib.utils import load_model, save_model

# Draw plots without opening a window
matplotlib.use("Agg")


# Small random datasets so the tests never download CIFAR10 or Fashion-MNIST
def fake_color_loader():
    images = torch.rand(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    return DataLoader(TensorDataset(images, labels), batch_size=8)


def fake_gray_loader():
    images = torch.rand(16, 1, 32, 32)
    labels = torch.randint(0, 10, (16,))
    return DataLoader(TensorDataset(images, labels), batch_size=8)


@pytest.mark.parametrize("name", ["FCNN", "CNN", "EnhancedCNN"])
def test_classifiers_output_10_scores(name):
    model = get_model(name)
    model.eval()
    assert model(torch.rand(4, 3, 32, 32)).shape == (4, 10)


def test_cnn64_matches_assignment_spec():
    model = get_model("CNN64")
    assert model(torch.rand(2, 3, 64, 64)).shape == (2, 10)
    assert model.fc1.in_features == 32 * 16 * 16
    assert model.fc1.out_features == 100


def test_vae_output_shapes():
    recon, mu, logvar = get_model("VAE")(torch.rand(4, 1, 32, 32))
    assert recon.shape == (4, 1, 32, 32)
    assert mu.shape == (4, 2)
    assert logvar.shape == (4, 2)


def test_unknown_model():
    with pytest.raises(ValueError):
        get_model("ResNet")


def test_unknown_dataset():
    with pytest.raises(ValueError):
        get_data_loader("data", dataset="MNIST")


def test_train_and_evaluate():
    model = get_model("CNN")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trained = train_model(model, fake_color_loader(), criterion, optimizer, epochs=1)
    avg_loss, accuracy = evaluate_model(trained, fake_color_loader(), criterion)
    assert avg_loss > 0
    assert 0 <= accuracy <= 100


def test_train_vae_and_generate():
    vae = get_model("VAE")
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    trained = train_vae_model(vae, fake_gray_loader(), vae_loss_function, optimizer, epochs=1)
    generate_samples(trained, "cpu", num_samples=10)


def test_save_and_load(tmp_path):
    model = get_model("CNN")
    path = save_model(model, str(tmp_path / "cnn.pth"))
    loaded = load_model(get_model("CNN"), path)
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
