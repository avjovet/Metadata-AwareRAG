import os
from pathlib import Path
from typing import Optional


class Settings:
    """Configuración centralizada para el proyecto."""
    
    # Rutas de la base de datos
    CHROMA_PERSIST_PATH: str = os.getenv("CHROMA_PERSIST_PATH", "./vector_dbs")
    
    # Configuración de Ollama
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
    # Configuración de embeddings
    DEFAULT_EMBEDDING_MODEL: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Configuración de re-ranking
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
    RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "5"))
    
    # Configuración de recuperación
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "10"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
    
    # Configuración de self-querying
    ENABLE_SELF_QUERY: bool = os.getenv("ENABLE_SELF_QUERY", "true").lower() == "true"
    
    # Configuración de logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"


# Instancia global de configuración
settings = Settings()
