from langchain.embeddings import OpenAIEmbeddings
from ..config import get_settings

settings = get_settings()

class EmbeddingService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )

    def get_embeddings(self, texts: list[str]):
        return self.embeddings.embed_documents(texts)

    def get_query_embedding(self, text: str):
        return self.embeddings.embed_query(text)
