from langchain.document_loaders import DirectoryLoader, PDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..config import get_settings

settings = get_settings()

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def load_documents(self, directory_path: str):
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PDFLoader
        )
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_csv(self, file_path: str):
        loader = CSVLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
