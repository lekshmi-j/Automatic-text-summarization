
# Automatic Text Summarization

This project builds an NLP system that automatically generates concise summaries of long documents.

## Problem
Given a long article (news, Wikipedia, or blog), generate a shorter version that preserves the most important information.

## Scope
- Language: English
- Input: Long text documents
- Output: 3–5 sentence summaries
- Methods:
  - Extractive summarization
  - Graph-based ranking
  - Abstractive summarization using Transformers


## Dataset

We use the CNN/DailyMail dataset, a large-scale news summarization benchmark.

Each example contains:
- A news article (hundreds of words)
- A human-written summary (2–5 sentences)

This makes summarization a supervised sequence-to-sequence learning problem.


Summarization requires:
- Understanding document meaning
- Selecting important content
- Removing redundancy
- Generating fluent natural language

Unlike sentiment or topic classification, the output is free-form text, making evaluation and learning significantly harder.

## Preprocessing Pipeline

We perform sentence-level preprocessing:

1. Sentence tokenization  
2. Word tokenization  
3. Stopword removal  
4. Lemmatization  
5. Cleaning (punctuation, numbers)

This prepares each sentence as a unit for ranking and modeling in later summarization steps.

