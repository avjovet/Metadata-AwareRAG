"""Utilidades para trabajar con documentos de LangChain."""

from typing import List
from langchain_core.documents import Document


def documents_to_text(documents: List[Document]) -> str:
    """
    Convierte una lista de documentos en texto concatenado.
    
    Args:
        documents: Lista de documentos de LangChain
        
    Returns:
        Texto concatenado de todos los documentos separados por doble salto de línea
    """
    return "\n\n".join([doc.page_content for doc in documents])


# Alias para compatibilidad con código existente
docs_to_text = documents_to_text

