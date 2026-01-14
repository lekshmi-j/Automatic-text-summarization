from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    chunks = list(chunk_text(text))
    summaries = []

    for chunk in chunks:
        s = summarizer(chunk, max_length=150, min_length=60, do_sample=False)
        summaries.append(s[0]["summary_text"])

    return " ".join(summaries)
