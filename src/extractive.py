import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

    #Each cell (i,j) now means:

    #How similar sentence i is to sentence j

 #Text Rank   
#sentence similarity algorithm
def build_similarity_matrix(sentences):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)

    similarity_matrix = cosine_similarity(tfidf, tfidf)
    np.fill_diagonal(similarity_matrix, 0)   # no self-loops

    return similarity_matrix

#TextRank Algorithm - Page Rank on sentences
def textrank_scores(similarity_matrix, damping=0.85, max_iter=50):
    n = similarity_matrix.shape[0]
    scores = np.ones(n) / n

    for _ in range(max_iter):
        new_scores = np.ones(n) * (1 - damping)
        for i in range(n):
            for j in range(n):
                if similarity_matrix[j][i] != 0:
                    new_scores[i] += damping * (
                        similarity_matrix[j][i] / similarity_matrix[j].sum()
                    ) * scores[j]
        scores = new_scores

    return scores
