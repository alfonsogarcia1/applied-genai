import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    # Fully connected network for CIFAR10 (Module 3 Practical 1)
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Return raw scores, CrossEntropyLoss applies softmax itself
        return self.fc3(x)


class SimpleCNN(nn.Module):
    # Two conv layers (Module 3 Practical 3)
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves height and width
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class EnhancedCNN(nn.Module):
    # Four conv layers with BatchNorm, plus Dropout (Module 3 Practical 3)
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv -> BatchNorm -> ReLU -> Pool, four times: 32 -> 16 -> 8 -> 4 -> 2
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class CNN64(nn.Module):
    # Assignment 2 architecture, takes 3 x 64 x 64 images
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Two pools halve 64 -> 32 -> 16, so 32 channels x 16 x 16 go into the first FC layer
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class VariationalEncoder(nn.Module):
    # Maps a 1 x 32 x 32 image to the mean and log variance of a normal distribution
    def __init__(self, latent_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    # Reverses the encoder: latent point -> 1 x 32 x 32 image
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.convtrans1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.convtrans1(x))
        x = F.relu(self.convtrans2(x))
        # Sigmoid keeps pixels between 0 and 1, which BCE needs
        return torch.sigmoid(self.convtrans3(x))


class VAE(nn.Module):
    # Variational autoencoder (Module 4 Practical VAE)
    def __init__(self, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        # Sample z = mu + eps * std so gradients can flow through mu and logvar
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class Generator(nn.Module):
    # Assignment 3 GAN generator: noise vector (batch, 100) -> 1 x 28 x 28 image
    def __init__(self, z_dim=100):
        super().__init__()
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)  # 14 -> 28

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.bn1(self.deconv1(x)))
        # Tanh gives pixels between -1 and 1, the same range the MNIST images are scaled to
        return torch.tanh(self.deconv2(x))


class Discriminator(nn.Module):
    # Assignment 3 GAN discriminator: 1 x 28 x 28 image -> one real/fake score
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)  # 28 -> 14
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 14 -> 7
        self.bn2 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = self.flatten(x)
        # Raw score, BCEWithLogitsLoss applies the sigmoid that turns it into a probability
        return self.fc(x)


class GAN(nn.Module):
    # Both networks in one model, so they save and load together
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()


class LSTMModel(nn.Module):
    # Word-level LSTM text generator (Module 6 Practical RNN)
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x is a batch of word ids, shape (batch, seq_len).
        # Returns a score for every vocabulary word at every position, plus the hidden state
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        return self.fc(x), hidden


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    # diffusion_times in [0, 1]: 0 is the clean image, 1 is (almost) pure noise.
    # signal_rate^2 + noise_rate^2 = 1, so the noisy image keeps variance 1
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    noise_rates = torch.sin(diffusion_angles)
    signal_rates = torch.cos(diffusion_angles)
    return noise_rates, signal_rates


class SinusoidalEmbedding(nn.Module):
    # Turns the noise variance (one number per image) into a vector of sines and cosines
    def __init__(self, num_frequencies=16):
        super().__init__()
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        # x: (batch, 1, 1, 1) -> (batch, 1, 1, 2 * num_frequencies)
        return torch.cat([torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # A 1x1 conv makes the shortcut match the output channels when they differ
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.proj(x)
        x = F.silu(self.conv1(x))  # silu is the swish activation, x * sigmoid(x)
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    # Residual blocks, saving each output for the matching UpBlock, then halve the size
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    # Double the size, then residual blocks that also take the saved DownBlock outputs
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    # Predicts the noise in a noisy image, given how much noise was added (Module 8 Practical)
    def __init__(self, image_size=64, num_channels=3):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=16)

        self.down1 = DownBlock(32, block_depth=2, in_channels=64)  # 32 image + 32 noise channels
        self.down2 = DownBlock(64, block_depth=2, in_channels=32)
        self.down3 = DownBlock(96, block_depth=2, in_channels=64)

        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)

        self.up1 = UpBlock(96, block_depth=2, in_channels=128)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)  # start by predicting zero noise

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        # Spread the noise embedding over every pixel and stack it with the image channels
        noise_emb = self.embedding(noise_variances).permute(0, 3, 1, 2)
        noise_emb = F.interpolate(noise_emb, size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


class DiffusionModel(nn.Module):
    # UNet plus the noise schedule, an EMA copy for generating, and the image mean and std
    def __init__(self, image_size=64, num_channels=3, ema_decay=0.8):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.network = UNet(image_size, num_channels)
        # Running average of the network weights, used for generating
        self.ema_network = copy.deepcopy(self.network)
        self.ema_decay = ema_decay
        self.schedule = offset_cosine_diffusion_schedule
        # Buffers so they are saved and loaded together with the weights
        self.register_buffer("normalizer_mean", torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("normalizer_std", torch.ones(1, num_channels, 1, 1))

    def set_normalizer(self, mean, std):
        self.normalizer_mean.copy_(mean.view(1, -1, 1, 1))
        self.normalizer_std.copy_(std.view(1, -1, 1, 1))

    def normalize(self, images):
        return (images - self.normalizer_mean) / self.normalizer_std

    def denormalize(self, images):
        return torch.clamp(images * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        network = self.network if training else self.ema_network
        pred_noises = network(noisy_images, noise_rates ** 2)
        # Undo the noising formula noisy = signal * image + noise * noise to estimate the image
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1.0 - self.ema_decay) * param)

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # Start from pure noise and step from t = 1 back to t = 0: at each step predict the
        # clean image, then re-noise it a little less than before
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.full((initial_noise.size(0), 1, 1, 1), 1.0 - step * step_size, device=initial_noise.device)
            noise_rates, signal_rates = self.schedule(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            next_noise_rates, next_signal_rates = self.schedule(t - step_size)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps=100):
        device = self.normalizer_mean.device
        initial_noise = torch.randn(num_images, self.num_channels, self.image_size, self.image_size, device=device)
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))


def get_model(model_name, **kwargs):
    # kwargs go to the model, e.g. get_model("LSTM", vocab_size=10000)
    if model_name == "FCNN":
        model = FCNN(**kwargs)
    elif model_name == "CNN":
        model = SimpleCNN(**kwargs)
    elif model_name == "EnhancedCNN":
        model = EnhancedCNN(**kwargs)
    elif model_name == "CNN64":
        model = CNN64(**kwargs)
    elif model_name == "VAE":
        model = VAE(**kwargs)
    elif model_name == "GAN":
        model = GAN(**kwargs)
    elif model_name == "LSTM":
        model = LSTMModel(**kwargs)
    elif model_name == "Diffusion":
        model = DiffusionModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model
