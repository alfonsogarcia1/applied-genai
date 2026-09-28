import spacy


class EmbeddingModel:
    def __init__(self, model_name="en_core_web_lg"):
        # Loading the model takes a second, so do it once at startup
        self.nlp = spacy.load(model_name)

    def has_vector(self, word):
        return self.nlp(word).has_vector

    def calculate_embedding(self, word):
        # For more than one word, spaCy averages the word vectors
        return self.nlp(word).vector.tolist()

    def calculate_similarity(self, word1, word2):
        return float(self.nlp(word1).similarity(self.nlp(word2)))
