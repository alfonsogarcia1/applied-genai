from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_embedding_returns_300_numbers():
    response = client.post("/embedding", json={"word": "apple"})
    assert response.status_code == 200
    data = response.json()
    assert data["word"] == "apple"
    assert data["dimension"] == 300
    assert len(data["embedding"]) == 300


def test_embedding_unknown_word():
    response = client.post("/embedding", json={"word": "qwzxvbn"})
    assert response.status_code == 404


def test_embedding_empty_word():
    response = client.post("/embedding", json={"word": ""})
    assert response.status_code == 422


def test_similar_words_score_higher():
    fruit = client.post("/similarity", json={"word1": "apple", "word2": "banana"})
    car = client.post("/similarity", json={"word1": "apple", "word2": "car"})
    assert fruit.status_code == 200
    assert fruit.json()["similarity"] > car.json()["similarity"]


def test_similarity_unknown_word():
    response = client.post("/similarity", json={"word1": "apple", "word2": "qwzxvbn"})
    assert response.status_code == 404


def test_generate_still_works():
    response = client.post("/generate", json={"start_word": "bigram", "length": 5})
    assert response.status_code == 200
    assert response.json()["generated_text"].startswith("bigram")
