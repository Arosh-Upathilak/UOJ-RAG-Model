import faiss
import numpy as np
from typing import List
import pickle
from pathlib import Path

class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = []

    def create_index(self, embeddings: List[List[float]], documents: List[str]):
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents = documents

    def save(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, directory: str):
        path = Path(directory)
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def search(self, query_embedding: List[float], k: int = 4):
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]
