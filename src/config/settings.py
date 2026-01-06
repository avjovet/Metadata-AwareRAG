import os
from pathlib import Path
from typing import Optional


class Settings:
    """Configuración centralizada para el proyecto."""
    
    # Rutas de datos y base de datos
    DATA_PATH: str = os.getenv("DATA_PATH", "data")
    CHROMA_PERSIST_PATH: str = os.getenv("CHROMA_PERSIST_PATH", "./vector_dbs")
    
    # Configuración de Ollama
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
    # Configuración de embeddings
    EMBEDDER_MODEL: str = os.getenv("EMBEDDER_MODEL", "BAAI/bge-m3")
    DEFAULT_EMBEDDING_MODEL: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Configuración de chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "30"))
    
    # Configuración de re-ranking
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
    RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "5"))
    
    # Configuración de recuperación
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "10"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
    MIN_DOCS_FOR_HYDE: int = int(os.getenv("MIN_DOCS_FOR_HYDE", "5"))  # Mínimo de docs para activar HYDE
    
    # Valores por defecto específicos de pipelines (mantener valores actuales)
    NAIVE_PIPELINE_TEMPERATURE: float = 0.1
    NAIVE_PIPELINE_TOP_K: int = 5
    DYNAMIC_PIPELINE_TOP_K: int = 15
    
    # Configuración de self-querying
    ENABLE_SELF_QUERY: bool = os.getenv("ENABLE_SELF_QUERY", "true").lower() == "true"
    
    # Configuración de logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # Configuración de base de datos vectorial
    DEFAULT_DB_IDENTIFIER: str = os.getenv("DEFAULT_DB_IDENTIFIER", "json_metadata")
    # Para bases de datos raw (con chunking), usar formato: db_{model}_cs{size}_co{overlap}
    # Para bases de datos JSON, usar formato: db_{model}_{identifier}


# Instancia global de configuración
settings = Settings()


def get_default_db_folder_name(
    embedding_model: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    db_identifier: Optional[str] = None
) -> str:
    """
    Construye el nombre de la carpeta de la base de datos vectorial.
    
    Args:
        embedding_model: Modelo de embeddings (por defecto desde settings)
        chunk_size: Tamaño de chunk (por defecto desde settings)
        chunk_overlap: Overlap de chunk (por defecto desde settings)
        db_identifier: Identificador para bases JSON (ej: "json_metadata")
        
    Returns:
        Nombre de la carpeta de la base de datos
    """
    model = embedding_model or settings.EMBEDDER_MODEL
    safe_model_name = model.replace("/", "_")
    
    if db_identifier:
        # Formato para JSON: db_{model}_{identifier}
        return f"db_{safe_model_name}_{db_identifier}"
    else:
        # Formato para raw: db_{model}_cs{chunk_size}_co{chunk_overlap}
        cs = chunk_size or settings.CHUNK_SIZE
        co = chunk_overlap or settings.CHUNK_OVERLAP
        return f"db_{safe_model_name}_cs{cs}_co{co}"
