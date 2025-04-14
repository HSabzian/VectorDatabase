
import numpy as np

def tokenize(sentence):
    """Tokenizes a sentence into words."""
    return sentence.lower().split()


def create_vocabulary(sentences):
    """Creates a vocabulary from the list of sentences."""
    vocabulary = set()
    for sentence in sentences:
        tokens = tokenize(sentence)
        vocabulary.update(tokens)
    return vocabulary


def vectorize_sentence(sentence, vocabulary, word_to_index):
    """Converts a sentence into a vector using the vocabulary."""
    tokens = tokenize(sentence)
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        if token in word_to_index:
            vector[word_to_index[token]] += 1
    return vector
