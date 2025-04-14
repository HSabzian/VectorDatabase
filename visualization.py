# visualization.py (for plotting)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_similar_sentences(similar_sentences_data, query_sentence):
    """Plots a bar chart of similar sentences."""
    sentences_list = [sentence for sentence, _ in similar_sentences_data]
    similarities = [similarity for _, similarity in similar_sentences_data]

    plt.figure(figsize=(20, 6))
    sns.barplot(x=similarities, y=sentences_list, palette="Blues_d")

    plt.title(f"Similar Sentences to: '{query_sentence}'", fontsize=16)
    plt.xlabel("Similarity Score", fontsize=14)
    plt.ylabel("Sentences", fontsize=1)
    plt.xlim(0, 1)

    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()

