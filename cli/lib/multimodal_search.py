from PIL import Image
from sentence_transformers import SentenceTransformer

from .semantic_search import cosine_similarity

class MultimodalSearch():
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self,image_path):
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0]
        return embedding

    def search_with_image(self,image_path,limit=5):
        embedding = self.embed_image(image_path)
        results = []

        for i, text_embedding in enumerate(self.text_embeddings):
            score = cosine_similarity(embedding, text_embedding)
            doc = self.documents[i]
            results.append({"id": doc["id"],
                            "title": doc["title"],
                            "description": doc["description"],
                            "score": score,
                            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

def verify_image_embedding(image_path):
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path,documents):
    searcher = MultimodalSearch(documents=documents)
    return searcher.search_with_image(image_path)

