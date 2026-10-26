import io

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from helper_lib.model import get_model
from helper_lib.utils import load_model


class GANGenerator:
    def __init__(self, weights_path="app/weights/gan_mnist.pth"):
        # Load the trained weights once at startup, only the generator is needed here
        gan = load_model(get_model("GAN"), weights_path)
        self.generator = gan.generator
        self.generator.eval()
        self.z_dim = gan.z_dim

    def generate_png(self, num_images=16):
        # Random noise -> digits, laid out in a grid and returned as PNG bytes
        with torch.no_grad():
            z = torch.randn(num_images, self.z_dim)
            images = (self.generator(z) + 1) / 2  # Tanh output [-1, 1] -> [0, 1]
        grid = make_grid(images, nrow=min(num_images, 8), padding=2)
        buffer = io.BytesIO()
        to_pil_image(grid).save(buffer, format="PNG")
        return buffer.getvalue()
