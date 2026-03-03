import json
import os
import numpy as np

from constants import *
from lib.semantic_search import *

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self,documents):
        self.documents = documents
        self.document_map = {}

        for movie in self.documents:
            self.document_map[movie["id"]] = movie
 
        all_chunks = []
        metadata = []
       
        for movie_idx,movie in enumerate(self.documents):
            if movie["description"] == "":
                continue
            chunks = semantic_chunk(movie["description"],4,1)
            
            for chunk_idx,chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({"movie_idx":movie_idx,"chunk_idx":chunk_idx,"total_chunks":len(chunks)})
    
        self.chunk_embeddings = self.model.encode(all_chunks,show_progress_bar=True)
        self.chunk_metadata=metadata
        np.save("cache/chunk_embeddings.npy",self.chunk_embeddings)
        with open("cache/chunk_metadata.json","w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self,documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}

        for movie in self.documents:
            self.document_map[movie["id"]] = movie

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json","r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self,query: str, limit:int=10):
        embed_query = self.generate_embedding(query)
        chunk_scores = []

        for idx,chunk in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(chunk,embed_query)
            metadata = self.chunk_metadata[idx]
            chunk_scores.append({"chunk_idx": metadata["chunk_idx"],"movie_idx": metadata["movie_idx"],"score":similarity})

        scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in scores or score > scores[movie_idx]:
                scores[movie_idx] = score

        sorted_movies = sorted(scores.items(),key=lambda x: x[1], reverse=True)
        top = sorted_movies[:limit]
        results = []
        for movie_idx,score in top:
            movie = self.documents[movie_idx]
            results.append({"id":movie["id"],
                           "title":movie["title"],
                           "document": movie["description"][:100],
                           "score": round(score,SCORE_PRECISION),
                           "metadata": movie.get("metadata",{})})
        return results
