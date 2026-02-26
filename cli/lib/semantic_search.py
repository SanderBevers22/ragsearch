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
    embeddings = searcher.load_or_create_embeddings(documents["movies"])
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    searcher = SemanticSearch()
    embedding = searcher.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2) 

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

    def search(self,query,limit):
        results = []

        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")

        embedding = self.generate_embedding(query)
        for idx,movie_embedding in enumerate(self.embeddings):
            results.append((cosine_similarity(embedding,movie_embedding),self.documents[idx]))
        sorted_scores = sorted(results,key=lambda x: x[0],reverse=True)
        top_result = sorted_scores[:limit]
        return [{"score": score, "title": doc["title"], "description":doc["description"]} for score,doc in top_result]
