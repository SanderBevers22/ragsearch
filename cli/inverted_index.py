import math
import os
import pickle
import string

from collections import Counter
from constants import *
from preprocessing import preprocess

class InvertedIndex():
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, text,stopwords=None):
        tokens = preprocess(text,stopwords)
        
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        if doc_id not in self.doc_lengths:
            self.doc_lengths[doc_id] = 0
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        
        total = 0
        for doc in self.doc_lengths:
            total += self.doc_lengths[doc]

        return total/len(self.doc_lengths)

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

        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths,f)

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)

            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            
            with open("cache/term_frequencies.pkl","rb") as f:
                self.term_frequencies = pickle.load(f)

            with open("cache/doc_lengths.pkl", "rb") as f:
                self.doc_lengths = pickle.load(f)

        except FileNotFoundError:
            raise FileNotFoundError("Index files not found. Run build first.")

    def get_tf(self, doc_id, term):
        tokens = preprocess(term)
        if len(tokens) != 1:
            raise ValueError("get_tf expects only one token")
        token = tokens[0]
        
        return self.term_frequencies.get(doc_id, {}).get(token,0)

    def get_bm25_idf(self,term:str)->float:
        tokens = preprocess(term)
        if len(tokens) != 1:
            raise ValueError("Expected single token")
        token = tokens[0]

        totaldocs = len(self.docmap)
        totalmatch = len(self.get_documents(token))
    
        idf = math.log((totaldocs-totalmatch+0.5)/(totalmatch + 0.5)+1)
        return idf

    def bm25_idf_command(self,term):

        self.load()
        return self.get_bm25_idf(term)
    
    def get_bm25_tf(self,doc_id,term,k1=BM25_K1,b=BM25_B):
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf = self.get_tf(doc_id,term)
        return ((tf*(k1+1))/(tf+k1*length_norm))

    def bm25_tf_command(self,doc_id,term,k1=BM25_K1,b=BM25_B):
        self.load()
        return self.get_bm25_tf(doc_id,term,k1,b)

    def bm25(self,doc_id,term):
        tf = self.get_bm25_tf(doc_id,term)
        idf = self.get_bm25_idf(term)
        return tf*idf

    def bm25_search(self,query,limit):
        tokens = preprocess(query)
        scores = {}
        for doc in self.docmap:
            total = 0
            for token in tokens:
                total += self.bm25(doc,token)

            scores[doc] = total

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]
