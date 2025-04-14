

from utils import tokenize, create_vocabulary, vectorize_sentence
from vector_store import VectorStore
from visualization import plot_similar_sentences
import numpy as np


#### 1:   Initialize VectorStore

vector_store = VectorStore()

#### 2:  External Knowledge Source

sentences = [
    # Neural Networks - fundamentals
    "A neural network is a computational model inspired by the human brain's structure and function.",
    "Artificial neurons process input signals by applying weights and activation functions.",
    "Feedforward neural networks pass information in one direction, from input to output layers.",
    "Backpropagation is a training algorithm that adjusts weights to minimize prediction error.",
    "The activation function introduces non-linearity to neural network computations.",

    # Architectures and types
    "Convolutional Neural Networks (CNNs) are widely used in image recognition tasks.",
    "Recurrent Neural Networks (RNNs) are suited for sequential data like time series and language.",
    "Long Short-Term Memory (LSTM) networks help RNNs remember long-range dependencies.",
    "Generative Adversarial Networks (GANs) consist of a generator and a discriminator working in opposition.",
    "Autoencoders compress and reconstruct data, often used for denoising or dimensionality reduction.",

    # Training and optimization
    "Stochastic Gradient Descent is a common optimization method used to train neural networks.",
    "Overfitting occurs when a neural network performs well on training data but poorly on unseen data.",
    "Regularization techniques like dropout and L2 help prevent overfitting in neural networks.",
    "Batch normalization accelerates training and improves performance by normalizing layer inputs.",
    "Learning rate determines the size of weight updates during training and affects convergence.",

    # Applications
    "Neural networks are used in natural language processing for tasks like translation and sentiment analysis.",
    "In autonomous vehicles, neural networks process sensor data to detect objects and make driving decisions.",
    "Medical image analysis uses CNNs to detect anomalies like tumors in X-rays and MRIs.",
    "Speech recognition systems use deep learning to convert spoken words into text.",
    "Neural networks can generate realistic images, music, or text using generative models.",

    # Advanced concepts
    "Transfer learning allows neural networks trained on one task to be adapted to another with less data.",
    "Attention mechanisms help neural networks focus on relevant parts of the input sequence.",
    "Transformers are deep learning models that outperform RNNs in many NLP tasks.",
    "Neural networks with many layers are referred to as deep neural networks or DNNs.",
    "Explainability in neural networks aims to make model decisions more interpretable to humans."
]

#### 3:  Create vocabulary

vocabulary = create_vocabulary(sentences)
word_to_index = {word: i for i, word in enumerate(vocabulary)}

#### 4: Vectorize sentences and store in VectorStore

for sentence in sentences:
    vector = vectorize_sentence(sentence, vocabulary, word_to_index)
    vector_store.add_vector(sentence, vector)

#### 5 :  Handle user query and perform similarity search

query_sentence = input("Your Query Here: ")
query_vector = vectorize_sentence(query_sentence, vocabulary, word_to_index)

similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=5)

#### 6:  Data Preparation for visualization

similar_sentences_data = [(sentence, similarity) for sentence, similarity in similar_sentences]

# Display similar sentences (sematics) and visulaization

print("Query Sentence:", query_sentence)
print("Similar Sentences:")
for sentence, similarity in similar_sentences_data:
    print(f"{sentence}: Similarity = {similarity:.4f}")

##  Visualize similar sentences

plot_similar_sentences(similar_sentences_data, query_sentence)
