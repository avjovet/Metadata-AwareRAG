"""Módulo de self-query con routing semántico y filtrado adaptativo."""

# Exportar funciones públicas principales
from .retrieval import (
    create_self_query_retriever,
    create_modular_self_query_pipeline
)

# Exportar funciones internas si se necesitan para testing o uso avanzado
from .routers import (
    create_semantic_router,
    create_filter_extractor
)
from .filter_validation import validate_and_normalize_filters
from .filter_strategies import create_filter_strategies
from .retrieval import build_chromadb_filter, create_retrieval_assembler

__all__ = [
    # Funciones principales (API pública)
    'create_self_query_retriever',
    'create_modular_self_query_pipeline',
    # Funciones internas (para uso avanzado)
    'create_semantic_router',
    'create_filter_extractor',
    'validate_and_normalize_filters',
    'create_filter_strategies',
    'build_chromadb_filter',
    'create_retrieval_assembler'
]

