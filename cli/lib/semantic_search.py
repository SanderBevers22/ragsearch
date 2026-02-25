import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer

def verify_model():
    searcher = SemanticSearch()
    print(f"Model loaded: {searcher.model}")
    print(f"Max sequence length: {searcher.model.max_seq_length}")

def embed_text(text):
    searcher = SemanticSearch()
    embed = searcher.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embed[:3]}")
    print(f"Dimensions: {embed.shape[0]}")

def verify_embeddings():
    searcher = SemanticSearch()
    
    with open("data/movies.json", "r") as f:
        documents = json.load(f)
    embeddings = searcher.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

    

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}


    def generate_embedding(self,text):
        if text == "" or text is None:
            raise ValueError("Text only contains whitespace or is empty")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self,documents):
        self.documents = documents
        self.document_map = {}
        all = []

        for movie in self.documents:
            self.document_map[movie["id"]] = movie
            all.append(f"{movie['title']}: {movie['description']}")
        self.embeddings = self.model.encode(all,show_progress_bar=True)

        np.save("cache/movie_embeddings.npy",self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self,documents):
        self.documents = documents
        self.document_map = {}

        for movie in self.documents:
            self.document_map[movie["id"]] = movie

        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")

            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)
