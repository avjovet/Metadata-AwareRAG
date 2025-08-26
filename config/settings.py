
class Settings:
    """Configuration settings for the RAG application."""
    
    DATA_PATH = "data"
    CHROMA_PERSIST_PATH = "vector_dbs"

    OLLAMA_MODEL = "llama3.1:8b"
    OLLAMA_URL = "http://localhost:11434"

    EMBEDDER_MODEL = "BAAI/bge-m3"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 30
    RETRIEVER_K = 5

settings = Settings()


 