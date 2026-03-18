import string

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def load_stopwords():
    with open("data/stopwords.txt","r") as f:
        return f.read().splitlines()

def preprocess(text,stopwords=None):
    if not stopwords:
        stopwords = load_stopwords()
    text = text.lower().translate(str.maketrans("","",string.punctuation))
    tokens = [t for t in text.split() if t]
    tokens = [t for t in tokens if t not in stopwords and t != ""]
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens
