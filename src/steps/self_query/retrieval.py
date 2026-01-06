"""Ensamblador de recuperación y pipeline modular de self-query."""

import logging
from typing import Any, Callable, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.language_models import BaseLanguageModel
from langchain_chroma import Chroma

from ..types import ExtractedFilters
from .filter_validation import validate_and_normalize_filters
from .filter_strategies import create_filter_strategies
from .routers import create_semantic_router, create_filter_extractor

logger = logging.getLogger(__name__)


def build_chromadb_filter(filters: Dict[str, Any]) -> Optional[dict]:
    """
    Construye un filtro de ChromaDB a partir de un diccionario de filtros.
    
    Args:
        filters: Diccionario con filtros a aplicar
        
    Returns:
        Filtro de ChromaDB en formato compatible, o None si no hay filtros
    """
    if not filters:
        return None
    
    filter_conditions = []
    
    for field, value in filters.items():
        if value is not None:
            if isinstance(value, int):
                filter_conditions.append({field: {"$eq": value}})
            elif isinstance(value, str):
                filter_conditions.append({field: {"$eq": value}})
            elif isinstance(value, list):
                filter_conditions.append({field: {"$in": value}})
    
    if len(filter_conditions) == 0:
        return None
    elif len(filter_conditions) == 1:
        return filter_conditions[0]
    else:
        return {"$and": filter_conditions}


def create_retrieval_assembler(vectorstore: Chroma, top_k: int = 15) -> Runnable:
    """
    Crea un ensamblador de recuperación que aplica estrategias de filtrado adaptativas.
    
    Args:
        vectorstore: Almacén vectorial de ChromaDB
        top_k: Número máximo de documentos a recuperar
        
    Returns:
        Runnable que recupera documentos usando estrategias de filtrado progresivas
    """
    def retrieval_assembler_step(inputs: Dict[str, Any]) -> List[Document]:
        question = inputs.get("question", "")
        filters = inputs.get("extracted_filters", ExtractedFilters())
        semantic_category = inputs.get("semantic_category", "general")
        
        try:
            validated_filters, discarded_filters = validate_and_normalize_filters(filters)
            
            strategies = create_filter_strategies(validated_filters, semantic_category)
            
            for strategy in strategies:
                chroma_filter = build_chromadb_filter(strategy['filters'])
                
                if chroma_filter:
                    retriever = vectorstore.as_retriever(
                        search_kwargs={"k": top_k, "filter": chroma_filter}
                    )
                else:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
                
                docs = retriever.invoke(question)
                
                if docs:
                    return docs
            
            return []
            
        except Exception as e:
            logger.error(f"Error en retrieval assembler: {e}", exc_info=True)
            try:
                basic_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
                docs = basic_retriever.invoke(question)
                return docs
            except Exception as e2:
                logger.error(f"Error en fallback retrieval: {e2}", exc_info=True)
                return []
    
    return RunnableLambda(retrieval_assembler_step)


def create_modular_self_query_pipeline(
    llm: BaseLanguageModel, 
    vectorstore: Chroma, 
    top_k: int = 15
) -> Dict[str, Runnable]:
    """
    Crea un pipeline modular de self-query con componentes separados.
    
    Args:
        llm: Modelo de lenguaje para routing y extracción
        vectorstore: Almacén vectorial de ChromaDB
        top_k: Número máximo de documentos a recuperar
        
    Returns:
        Diccionario con componentes: semantic_router, filter_extractor, retrieval_assembler
    """
    semantic_router = create_semantic_router(llm)
    filter_extractor = create_filter_extractor(llm)
    retrieval_assembler = create_retrieval_assembler(vectorstore, top_k)
    
    return {
        "semantic_router": semantic_router,
        "filter_extractor": filter_extractor,
        "retrieval_assembler": retrieval_assembler
    }


def create_self_query_retriever(
    llm: BaseLanguageModel, 
    vectorstore: Chroma, 
    top_k: int = 15
) -> Callable[[str], List[Document]]:
    """
    Crea un retriever simple de self-query como fallback.
    
    Args:
        llm: Modelo de lenguaje (no usado actualmente, para compatibilidad)
        vectorstore: Almacén vectorial de ChromaDB
        top_k: Número máximo de documentos a recuperar
        
    Returns:
        Función que recupera documentos usando búsqueda semántica simple
    """
    def simple_self_query_retriever(question: str) -> List[Document]:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(question)
            return docs
        except Exception as e:
            logger.error(f"Error en fallback retrieval: {e}", exc_info=True)
            return []
    
    return simple_self_query_retriever

