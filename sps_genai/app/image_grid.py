import io

from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


def to_png_grid(images, nrow=8):
    # A batch of images in [0, 1] -> one PNG with up to nrow images per row, 2 pixels apart
    grid = make_grid(images.clamp(0, 1), nrow=min(len(images), nrow), padding=2)
    buffer = io.BytesIO()
    to_pil_image(grid).save(buffer, format="PNG")
    return buffer.getvalue()
