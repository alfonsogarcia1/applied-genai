# Module 6 RNN activity: train a word-level LSTM on The Count of Monte Cristo
# and save the weights and vocabulary the API loads.
# Download the book first (from sps_genai/):
#   curl -L -o data/monte_cristo/pg1184.txt https://www.gutenberg.org/cache/epub/1184/pg1184.txt
# Then run:  uv run python -m scripts.train_rnn
import json

import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import build_vocab, get_text_data_loader, tokenize
from helper_lib.generator import generate_text
from helper_lib.model import get_model
from helper_lib.trainer import train_rnn_model
from helper_lib.utils import get_device, save_model

BOOK_PATH = "data/monte_cristo/pg1184.txt"
WEIGHTS_PATH = "app/weights/lstm_monte_cristo.pth"
VOCAB_PATH = "app/weights/lstm_vocab.json"
VOCAB_SIZE = 10000
EPOCHS = 3

torch.manual_seed(42)
device = get_device()

with open(BOOK_PATH, encoding="utf-8") as f:
    book = f.read()

# Keep only the novel, not the Project Gutenberg license text around it
start = book.find("\n", book.find("*** START OF")) + 1
end = book.find("*** END OF")
book = book[start:end]

vocab = build_vocab(tokenize(book), VOCAB_SIZE)
train_loader = get_text_data_loader(book, vocab, batch_size=64, seq_len=30)
print(f"{len(tokenize(book))} words, {len(train_loader.dataset)} training sequences")

model = get_model("LSTM", vocab_size=len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trained_model = train_rnn_model(model, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)

# Save on the CPU so the weights load anywhere, including the Docker container
save_model(trained_model.to("cpu"), WEIGHTS_PATH)
with open(VOCAB_PATH, "w") as f:
    json.dump(vocab, f)
print(f"Saved weights to {WEIGHTS_PATH} and vocabulary to {VOCAB_PATH}")

for word in ["the", "monte", "she"]:
    print(generate_text(trained_model, vocab, word, length=20, temperature=0.8))
