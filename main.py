
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.models import CoherenceModel

nltk.download("stopwords")
stop_words = stopwords.words("english")

# 1. Load dataset (subset of categories to keep runtime light)
newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'),
                                categories=['sci.space', 'rec.sport.baseball', 'comp.graphics'],
                                shuffle=True, random_state=42)
texts = newsgroups.data

# 2. Preprocessing: tokenization, lowercasing, stopword removal
processed_texts = []
for doc in texts:
    tokens = [word.lower() for word in nltk.word_tokenize(doc) if word.isalpha()]
    tokens = [t for t in tokens if t not in stop_words and len(t) > 3]
    processed_texts.append(tokens)

# 3. Create dictionary and corpus for Gensim
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# 4. Train LDA with different topic numbers and compare coherence
topic_nums = [2, 3, 4, 5, 6]
coherence_scores = []

for k in topic_nums:
    lda_model = models.LdaModel(corpus, num_topics=k, id2word=dictionary, passes=10, random_state=42)
    coherence_model = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
    coherence_scores.append(coherence_model.get_coherence())
    print(f"\n--- Top Words for {k} Topics ---")
    for idx, topic in lda_model.show_topics(num_topics=k, formatted=False):
        print(f"Topic {idx}: {[word for word, _ in topic]}")

# 5. Plot coherence vs number of topics
plt.figure(figsize=(6,4))
plt.plot(topic_nums, coherence_scores, marker="o")
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.title("Coherence Scores for LDA Topic Models")
plt.show()
