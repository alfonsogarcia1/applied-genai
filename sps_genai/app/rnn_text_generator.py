import json

from helper_lib.generator import generate_text
from helper_lib.model import get_model
from helper_lib.utils import load_model


class RNNTextGenerator:
    def __init__(self, weights_path="app/weights/lstm_monte_cristo.pth",
                 vocab_path="app/weights/lstm_vocab.json"):
        # Load the vocabulary and trained weights once at startup
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.known_words = set(self.vocab[1:])  # skip the <unk> placeholder
        self.model = load_model(get_model("LSTM", vocab_size=len(self.vocab)), weights_path)
        self.model.eval()

    def knows(self, word):
        return word.lower() in self.known_words

    def generate(self, start_word, length=20, temperature=0.8):
        return generate_text(self.model, self.vocab, start_word.lower(), length, temperature)
