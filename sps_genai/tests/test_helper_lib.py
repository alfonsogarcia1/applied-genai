import matplotlib
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from helper_lib.data_loader import (
    UNKNOWN,
    TextDataset,
    build_vocab,
    get_data_loader,
    get_text_data_loader,
    tokenize,
)
from helper_lib.evaluator import evaluate_model
from helper_lib.generator import generate_samples, generate_text
from helper_lib.model import get_model
from helper_lib.trainer import (
    train_diffusion,
    train_gan,
    train_model,
    train_rnn_model,
    train_vae_model,
    vae_loss_function,
)
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
        get_data_loader("data", dataset="ImageNet")


def test_gan_matches_assignment_spec():
    gan = get_model("GAN")
    gen, disc = gan.generator, gan.discriminator
    z = torch.randn(4, 100)

    # Generator: 100 -> 7x7x128 -> 64 x 14 x 14 -> 1 x 28 x 28 in [-1, 1]
    x = gen.fc(z).view(-1, 128, 7, 7)
    assert gen.deconv1(x).shape == (4, 64, 14, 14)
    images = gen(z)
    assert images.shape == (4, 1, 28, 28)
    assert images.min() >= -1 and images.max() <= 1

    # Discriminator: 1 x 28 x 28 -> 64 x 14 x 14 -> 128 x 7 x 7 -> 1 score
    h = disc.conv1(images)
    assert h.shape == (4, 64, 14, 14)
    assert disc.conv2(h).shape == (4, 128, 7, 7)
    assert disc(images).shape == (4, 1)


def test_train_gan_and_generate():
    gan = get_model("GAN")
    images = torch.rand(16, 1, 28, 28) * 2 - 1
    loader = DataLoader(TensorDataset(images, torch.zeros(16)), batch_size=8)
    optimizer = {
        "generator": optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        "discriminator": optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
    }
    before = gan.generator.fc.weight.clone()
    trained = train_gan(gan, loader, nn.BCEWithLogitsLoss(), optimizer, epochs=1)
    assert not torch.equal(before, trained.generator.fc.weight)
    generate_samples(trained, "cpu", num_samples=4)


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


TEXT = "The cat sat on the mat. The dog sat on the log. " * 10


def test_tokenize_and_vocab():
    assert tokenize("The Cat, sat!") == ["the", "cat", "sat"]
    vocab = build_vocab(tokenize(TEXT), max_size=4)
    # <unk> first, then the 3 most frequent words
    assert vocab == [UNKNOWN, "the", "sat", "on"]


def test_text_dataset_targets_are_shifted_by_one():
    dataset = TextDataset([0, 1, 2, 3, 4, 5], seq_len=3)
    assert len(dataset) == 3
    x, y = dataset[1]
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [2, 3, 4]


def test_lstm_output_shape():
    scores, (h, c) = get_model("LSTM", vocab_size=50)(torch.randint(0, 50, (4, 7)))
    assert scores.shape == (4, 7, 50)
    assert h.shape == (1, 4, 128)


def test_train_rnn_and_generate_text():
    vocab = build_vocab(tokenize(TEXT))
    loader = get_text_data_loader(TEXT, vocab, batch_size=8, seq_len=5)
    model = get_model("LSTM", vocab_size=len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    trained = train_rnn_model(model, loader, nn.CrossEntropyLoss(), optimizer, epochs=1)
    text = generate_text(trained, vocab, "the", length=8)
    words = text.split()
    assert len(words) == 8
    assert words[0] == "the"
    assert UNKNOWN not in words


def test_unet_predicts_noise_of_the_same_shape():
    unet = get_model("Diffusion").network
    noise = unet(torch.rand(2, 3, 64, 64), torch.rand(2, 1, 1, 1))
    assert noise.shape == (2, 3, 64, 64)


def test_train_diffusion_updates_ema_and_generates(tmp_path):
    # 32 x 32 keeps the test fast; the UNet halves the size three times, so any multiple of 8 works
    model = get_model("Diffusion", image_size=32)
    loader = DataLoader(TensorDataset(torch.rand(8, 3, 32, 32), torch.zeros(8)), batch_size=4)
    optimizer = optim.AdamW(model.network.parameters(), lr=1e-3)
    trained = train_diffusion(model, loader, nn.L1Loss(), optimizer, epochs=1)

    # The EMA copy should have moved towards the trained network, but not equal it
    live = trained.network.final.weight
    ema = trained.ema_network.final.weight
    assert not torch.equal(live, ema)
    assert ema.abs().sum() > 0

    images = trained.generate(num_images=2, diffusion_steps=5)
    assert images.shape == (2, 3, 32, 32)
    assert images.min() >= 0 and images.max() <= 1
    generate_samples(trained, "cpu", num_samples=2, diffusion_steps=5)

    # The normalizer is saved with the weights
    trained.set_normalizer(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.5, 0.5, 0.5]))
    path = save_model(trained, str(tmp_path / "diffusion.pth"))
    loaded = load_model(get_model("Diffusion", image_size=32), path)
    assert torch.allclose(loaded.normalizer_mean.flatten(), torch.tensor([0.1, 0.2, 0.3]))


def test_save_and_load(tmp_path):
    model = get_model("CNN")
    path = save_model(model, str(tmp_path / "cnn.pth"))
    loaded = load_model(get_model("CNN"), path)
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
