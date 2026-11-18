from app.image_grid import to_png_grid
from helper_lib.model import get_model
from helper_lib.utils import load_model


class DiffusionGenerator:
    def __init__(self, weights_path="app/weights/diffusion_cifar10.pth"):
        # Load the trained weights once at startup (UNet, its EMA copy and the image mean/std)
        self.model = load_model(get_model("Diffusion", image_size=32, num_channels=3), weights_path)
        self.model.eval()

    def generate_png(self, num_images=16, diffusion_steps=20):
        # Pure noise -> reverse diffusion -> CIFAR10-like images, as one PNG grid
        images = self.model.generate(num_images, diffusion_steps)
        return to_png_grid(images)
