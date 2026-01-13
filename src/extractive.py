import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#This gives each sentence a numerical importance score.
def tfidf_sentence_scores(cleaned_sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

    # Sum TF-IDF scores of each sentence
    scores = tfidf_matrix.sum(axis=1)
    scores = np.array(scores).flatten()

    return scores

def get_top_sentences(original_sentences, scores, k=5):
    ranked = list(enumerate(scores))
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, _ in ranked[:k]]
    top_indices.sort()   # preserve original order

    summary = [original_sentences[i] for i in top_indices]
    return " ".join(summary)
