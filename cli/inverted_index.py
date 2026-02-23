import os
import pickle
import string

from collections import Counter
from preprocessing import preprocess

class InvertedIndex():
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id, text,stopwords=None):
        tokens = preprocess(text,stopwords)
        
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            self.term_frequencies[doc_id][token] += 1
 
    def get_documents(self, term):
        term = term.lower()
        docs = self.index.get(term,set())
        return sorted(list(docs))

    def build(self,movies,stopwords=None):
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie

            combined = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id,combined,stopwords)

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index,f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap,f)

        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies,f)

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)

            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            
            with open("cache/term_frequencies.pkl","rb") as f:
                self.term_frequencies = pickle.load(f)

        except FileNotFoundError:
            raise FileNotFoundError("Index files not found. Run build first.")

    def get_tf(self, doc_id, term):
        tokens = preprocess(term)
        if len(tokens) != 1:
            raise ValueError("get_tf expects only one token")
        token = tokens[0]
        
        return self.term_frequencies.get(doc_id, {}).get(token,0)
