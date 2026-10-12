# sps_genai

FastAPI app for Applied Generative AI. It has a bigram text generator (Module 3),
word embeddings from spaCy's `en_core_web_lg` model (Assignment 1), and a CNN image
classifier trained on CIFAR10 (Assignment 2).

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8001:80 sps-genai
```

The API is then at http://localhost:8001 and the interactive docs are at
http://localhost:8001/docs.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | | hello world |
| POST | `/generate` | `{"start_word": "bigram", "length": 10}` | generated text |
| POST | `/embedding` | `{"word": "apple"}` | the 300-dimension vector for the word |
| POST | `/similarity` | `{"word1": "apple", "word2": "banana"}` | cosine similarity of the two words |

| POST | `/classify` | an image file (form field `file`) | the CIFAR10 class, confidence and all 10 probabilities |

`/embedding` and `/similarity` return 404 if a word isn't in the model's vocabulary.
`/classify` returns 400 if the file isn't an image.

```bash
curl -X POST http://localhost:8001/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'
```

```bash
curl -X POST http://localhost:8001/classify -F "file=@cat.jpg"
```

## Image classifier (Assignment 2)

`helper_lib/` holds the reusable PyTorch code: data loading, models, training and evaluation.
The classifier is `CNN64` in `helper_lib/model.py`:

| Layer | Output shape |
|---|---|
| Input RGB image | 3 x 64 x 64 |
| Conv2d 16 filters, 3x3, stride 1, padding 1 + ReLU | 16 x 64 x 64 |
| MaxPool 2x2, stride 2 | 16 x 32 x 32 |
| Conv2d 32 filters, 3x3, stride 1, padding 1 + ReLU | 32 x 32 x 32 |
| MaxPool 2x2, stride 2 | 32 x 16 x 16 |
| Flatten | 8192 |
| Linear + ReLU | 100 |
| Linear | 10 (one per CIFAR10 class) |

CIFAR10 images are 32 x 32, so they are resized to 64 x 64 to fit the architecture.
After 10 epochs with Adam (lr 0.001, batch size 64) it gets 65.95% accuracy on the
10,000 test images (81% on the training set).
The trained weights are in `app/weights/cnn64_cifar10.pth`, so Docker doesn't need to retrain.
To train again (downloads CIFAR10 into `data/` the first time):

```bash
uv run python -m scripts.train_classifier
```

## Run locally

```bash
uv sync
uv run fastapi dev app/main.py --port 8001
uv run pytest
```
