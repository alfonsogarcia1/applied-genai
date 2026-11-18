from app.image_grid import to_png_grid
from helper_lib.generator import generate_ebm_images
from helper_lib.model import get_model
from helper_lib.utils import load_model


class EBMGenerator:
    def __init__(self, weights_path="app/weights/ebm_cifar10.pth"):
        # Load the trained energy model once at startup
        self.model = load_model(get_model("EBM", num_channels=3, image_size=32), weights_path)
        self.model.eval()

    def generate_png(self, num_images=16, steps=256):
        # Uniform noise -> Langevin dynamics down the energy surface -> one PNG grid
        images = generate_ebm_images(self.model, num_images, steps=steps)
        return to_png_grid(images)
