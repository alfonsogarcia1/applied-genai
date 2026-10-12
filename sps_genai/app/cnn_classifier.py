import io

import torch
from PIL import Image
from torchvision import transforms

from helper_lib.model import get_model
from helper_lib.utils import load_model

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class CNNClassifier:
    def __init__(self, weights_path="app/weights/cnn64_cifar10.pth"):
        # Load the trained weights once at startup
        self.model = load_model(get_model("CNN64"), weights_path)
        self.model.eval()
        # The model was trained on 32 x 32 CIFAR10 images scaled up to 64 x 64,
        # so shrink uploads to 32 first to make them look like the training data
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.transform(image).unsqueeze(0)  # add the batch dimension
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        best = int(probs.argmax())
        return {
            "predicted_class": CLASSES[best],
            "confidence": float(probs[best]),
            "probabilities": {name: float(p) for name, p in zip(CLASSES, probs)},
        }
