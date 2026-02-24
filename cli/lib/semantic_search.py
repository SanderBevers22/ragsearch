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

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self,text):
        if text == "" or text is None:
            raise ValueError("Text only contains whitespace or is empty")
        embedding = self.model.encode(list(text))
        return embedding[0]
