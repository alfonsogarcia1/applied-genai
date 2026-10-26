from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.cnn_classifier import CNNClassifier
from app.embedding_model import EmbeddingModel
from app.gan_generator import GANGenerator
from app.rnn_text_generator import RNNTextGenerator

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()
cnn_classifier = CNNClassifier()
rnn_generator = RNNTextGenerator()
gan_generator = GANGenerator()

MAX_RNN_LENGTH = 500


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str = Field(min_length=1)


class SimilarityRequest(BaseModel):
    word1: str = Field(min_length=1)
    word2: str = Field(min_length=1)


def check_known(word):
    # spaCy gives unknown words a vector of all zeros, so reject them instead
    if not embedding_model.has_vector(word):
        raise HTTPException(status_code=404, detail=f"No embedding found for '{word}'")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    if not 1 <= request.length <= MAX_RNN_LENGTH:
        raise HTTPException(status_code=422, detail=f"length must be between 1 and {MAX_RNN_LENGTH}")
    # The LSTM can only start from a word it saw during training
    if not rnn_generator.knows(request.start_word):
        raise HTTPException(status_code=404, detail=f"'{request.start_word}' is not in the RNN vocabulary")
    generated_text = rnn_generator.generate(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    check_known(request.word)
    embedding = embedding_model.calculate_embedding(request.word)
    return {"word": request.word, "dimension": len(embedding), "embedding": embedding}


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    check_known(request.word1)
    check_known(request.word2)
    similarity = embedding_model.calculate_similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity": similarity}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        result = cnn_classifier.predict(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not an image")
    return {"filename": file.filename, **result}


@app.get("/generate_gan", response_class=Response, responses={200: {"content": {"image/png": {}}}})
def generate_gan(num_images: int = Query(16, ge=1, le=64)):
    # New hand-written digits from the GAN trained on MNIST, as one PNG grid
    png = gan_generator.generate_png(num_images)
    return Response(content=png, media_type="image/png")
