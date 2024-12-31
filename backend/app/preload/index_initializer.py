from app.services.document_loader import DocumentLoader
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
import os

def initialize_index(data_dir: str, persistent_dir: str):

    # Initialize services
    document_loader = DocumentLoader()
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    # Paths for the FAISS index and document storage
    index_path = os.path.join(persistent_dir, "index.faiss")
    documents_path = os.path.join(persistent_dir, "documents.pkl")

    if os.path.exists(index_path) and os.path.exists(documents_path):
        print("Index and documents already exist. Loading from disk...")
        vector_store.load(persistent_dir)
    else:
        print("Index and documents not found. Creating new index...")

        # Load documents from the data directory
        documents = document_loader.load_documents(data_dir)

        # Extract content for embedding
        document_texts = [doc.page_content for doc in documents]

        # Generate embeddings
        embeddings = embedding_service.get_embeddings(document_texts)

        # Create and save the index
        vector_store.create_index(embeddings, document_texts)
        vector_store.save(persistent_dir)

        print("Index created and saved successfully.")

if __name__ == "__main__":
    # Directory containing your preloaded documents (e.g., PDFs or text files)
    DATA_DIR = "./app/preload/documents"

    # Directory for persistent storage of FAISS index and documents
    PERSISTENT_DIR = "./persistent_data"

    # Ensure directories exist
    os.makedirs(PERSISTENT_DIR, exist_ok=True)

    # Initialize the index
    initialize_index(DATA_DIR, PERSISTENT_DIR)
