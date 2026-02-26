import json
import os
import numpy as np

from semantic_search import *

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self,documents):
        self.documents = documents
        self.document_map = {}

        chunks = []
        metadata = {}
        
        for movie in self.documents:
            if movie["text"] == "":
                continue
            #do stuff
        self.chunk_embeddings = self.model.encode(chunks,show_progress_bar=True)
        np.save("cache/chunk_embeddings.npy",self.chunk_embeddings)
        json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, "cache/chunk_metadata.json", indent=2)
        
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self,documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            self.chunk_metadata = json.load("cache/chunk_metadata.json")
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)
