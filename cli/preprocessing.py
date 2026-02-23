import string

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def preprocess(text,stopwords=None):
    text = text.lower().translate(str.maketrans("","",string.punctuation))
    tokens = [t for t in text.split() if t]

    if stopwords:
        tokens = [t for t in tokens if t not in stopwords and t != ""]

    tokens = [stemmer.stem(t) for t in tokens]

    return tokens
