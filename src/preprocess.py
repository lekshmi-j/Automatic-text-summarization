import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_sentence(sentence):
    # lowercase
    sentence = sentence.lower()
    
    # remove special characters and numbers
    sentence = re.sub(r"[^a-zA-Z\s]", "", sentence)
    
    # tokenize words
    words = word_tokenize(sentence)
    
    # remove stopwords + lemmatize
    cleaned = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]
    
    return " ".join(cleaned)


def preprocess_article(article):
    from nltk.tokenize import sent_tokenize
    
    sentences = sent_tokenize(article)
    cleaned_sentences = [clean_sentence(s) for s in sentences]
    
    return sentences, cleaned_sentences
