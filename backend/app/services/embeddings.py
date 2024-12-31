from sentence_transformers import SentenceTransformer
from app.config import get_settings

settings = get_settings()

class EmbeddingService:
    def __init__(self):

        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
    def get_embeddings(self, texts: list[str]):

        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()  # Convert numpy arrays to lists for consistency

    def get_query_embedding(self, text: str):

        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()  # Convert numpy array to list for consistency