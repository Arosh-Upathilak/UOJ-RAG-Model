from app.services.vector_store import VectorStore

# Initialize vector store
vector_store_instance = VectorStore()
vector_store_instance.load("persistent_data")  
def get_vector_store():
    return vector_store_instance
