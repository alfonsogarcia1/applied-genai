# sps_genai

FastAPI app for Applied Generative AI. It has a bigram text generator (Module 3) and
word embeddings from spaCy's `en_core_web_lg` model (Assignment 1).

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

`/embedding` and `/similarity` return 404 if a word isn't in the model's vocabulary.

```bash
curl -X POST http://localhost:8001/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'
```

## Run locally

```bash
uv sync
uv run fastapi dev app/main.py --port 8001
uv run pytest
```
