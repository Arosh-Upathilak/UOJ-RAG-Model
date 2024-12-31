from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    HUGGINGFACE_API_KEY: str

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    MODEL_NAME: str = "google/flan-t5-base"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    MAX_LENGTH: int = 512
    TOP_K: int = 50
    TOP_P: float = 0.95
    TEMPERATURE: float = 0.7

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
