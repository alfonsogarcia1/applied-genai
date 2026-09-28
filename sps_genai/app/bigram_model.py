import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        # Train once when the model is created, so requests are fast
        self.frequency_threshold = frequency_threshold
        self.vocab, self.bigram_probs = self._analyze_bigrams(" ".join(corpus))

    def _tokenize(self, text):
        # Lowercase everything and keep only whole words
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not self.frequency_threshold:
            return tokens
        # Drop rare words to keep the vocabulary small
        word_counts = Counter(tokens)
        return [t for t in tokens if word_counts[t] >= self.frequency_threshold]

    def _analyze_bigrams(self, text):
        words = self._tokenize(text)

        # Count each word, and each pair of words that appear next to each other
        bigram_counts = Counter(zip(words[:-1], words[1:]))
        unigram_counts = Counter(words)

        # Chance of word2 following word1 = how often the pair shows up / how often word1 shows up
        bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]

        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word, length=20):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)
            # Nothing ever follows this word, so stop early
            if not next_words:
                break
            # Pick the next word at random, but favour the more likely ones
            current_word = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            generated_words.append(current_word)

        return " ".join(generated_words)
