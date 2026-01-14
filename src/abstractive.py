from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    chunks = list(chunk_text(text))
    summaries = []

    for chunk in chunks:
        s = summarizer(chunk, max_length=150, min_length=60, do_sample=False)
        summaries.append(s[0]["summary_text"])

    return " ".join(summaries)

def chunk_text(text, max_words=500):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def summarize_article(article_text, max_length=150, min_length=60):
    chunks = chunk_text(article_text)
    summaries = []

    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        summaries.append(summary[0]["summary_text"])

    final_summary = " ".join(summaries)
    return final_summary