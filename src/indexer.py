# src/indexer.py

from typing import Optional
from .indexing_logic import index_raw_documents


def create_knowledge_base(
    embedding_model=None,
    embedding_model_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    force_reindex: bool = False
):
    """
    Función principal para crear la base de conocimiento.
    Esta función es llamada desde index.py
    """
    index_raw_documents(
        embedding_model_name=embedding_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_reindex=force_reindex
    )


if __name__ == "__main__":
    create_knowledge_base()
