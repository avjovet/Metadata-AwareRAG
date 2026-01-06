"""Estrategias de filtrado adaptativas para recuperación de documentos."""

from typing import Any, Dict, List, Optional


def _get_fixed_metadata(semantic_category: str) -> Dict[str, Any]:
    """
    Obtiene metadatos fijos basados en la categoría semántica.
    
    Args:
        semantic_category: Categoría semántica determinada por el router
        
    Returns:
        Diccionario con metadatos fijos para la categoría
    """
    fixed_metadata = {
        "constitucion": {
            "source": "Constitución Política del Perú",
            "document_type": "constitucion", 
            "topic": "derechos_fundamentales"
        },
        "derecho_laboral": {
            "source": "Compendio Derecho Laboral",
            "document_type": "decreto",
            "topic": "derecho_laboral"
        },
        "faq": {
            "source": "Preguntas Frecuentes", 
            "document_type": "faq",
            "topic": "Preguntas Frecuentes"
        },
        "general": {}
    }
    return fixed_metadata.get(semantic_category, {})


def _create_all_filters_strategy(
    fixed_filters: Dict[str, Any], 
    validated_filters: Dict[str, Any]
) -> Dict[str, Any]:
    """Crea estrategia con todos los filtros disponibles."""
    all_filters = fixed_filters.copy()
    for key, value in validated_filters.items():
        if value is not None:
            all_filters[key] = value
    
    return {
        "name": "todos_filtros",
        "filters": all_filters,
        "description": f"Todos los filtros: {list(all_filters.keys())}"
    }


def _create_without_title_strategy(
    fixed_filters: Dict[str, Any], 
    validated_filters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Crea estrategia sin el campo title."""
    filters_without_title = fixed_filters.copy()
    for key, value in validated_filters.items():
        if value is not None and key != 'title':
            filters_without_title[key] = value
    
    if len(filters_without_title) > len(fixed_filters):
        return {
            "name": "sin_title",
            "filters": filters_without_title,
            "description": f"Sin title: {list(filters_without_title.keys())}"
        }
    return None


def _create_without_doc_type_strategy(
    fixed_filters: Dict[str, Any], 
    validated_filters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Crea estrategia sin el campo document_type."""
    filters_without_doc_type = fixed_filters.copy()
    if 'document_type' in filters_without_doc_type:
        del filters_without_doc_type['document_type']
    for key, value in validated_filters.items():
        if value is not None and key not in ['title', 'document_type']:
            filters_without_doc_type[key] = value
    
    if len(filters_without_doc_type) > 0:
        return {
            "name": "sin_document_type",
            "filters": filters_without_doc_type,
            "description": f"Sin document_type: {list(filters_without_doc_type.keys())}"
        }
    return None


def _create_without_year_strategy(
    fixed_filters: Dict[str, Any], 
    validated_filters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Crea estrategia sin los campos document_type y year."""
    filters_without_year = fixed_filters.copy()
    for key in ['document_type', 'year']:
        if key in filters_without_year:
            del filters_without_year[key]
    for key, value in validated_filters.items():
        if value is not None and key not in ['title', 'document_type', 'year']:
            filters_without_year[key] = value
    
    if len(filters_without_year) > 0:
        return {
            "name": "sin_year",
            "filters": filters_without_year,
            "description": f"Sin year: {list(filters_without_year.keys())}"
        }
    return None


def _create_basic_filters_strategy(fixed_filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Crea estrategia con solo filtros básicos (source y topic)."""
    basic_filters = {}
    for key in ['source', 'topic']:
        if key in fixed_filters:
            basic_filters[key] = fixed_filters[key]
    
    if basic_filters:
        return {
            "name": "solo_basicos",
            "filters": basic_filters,
            "description": f"Solo básicos: {list(basic_filters.keys())}"
        }
    return None


def _create_no_filters_strategy() -> Dict[str, Any]:
    """Crea estrategia sin filtros (búsqueda semántica pura)."""
    return {
        "name": "sin_filtros",
        "filters": {},
        "description": "Búsqueda semántica pura en toda la BD"
    }


def create_filter_strategies(validated_filters: Dict[str, Any], semantic_category: str) -> List[Dict[str, Any]]:
    """
    Crea lista de estrategias de filtrado ordenadas por especificidad.
    
    Args:
        validated_filters: Filtros validados extraídos de la pregunta
        semantic_category: Categoría semántica determinada por el router
        
    Returns:
        Lista de estrategias de filtrado, desde más específica a menos específica
    """
    strategies = []
    fixed_filters = _get_fixed_metadata(semantic_category)
    has_variables = any(value is not None for value in validated_filters.values())
    
    if has_variables and fixed_filters:
        strategies.append(_create_all_filters_strategy(fixed_filters, validated_filters))
        
        without_title = _create_without_title_strategy(fixed_filters, validated_filters)
        if without_title:
            strategies.append(without_title)
        
        without_doc_type = _create_without_doc_type_strategy(fixed_filters, validated_filters)
        if without_doc_type:
            strategies.append(without_doc_type)
        
        without_year = _create_without_year_strategy(fixed_filters, validated_filters)
        if without_year:
            strategies.append(without_year)
    
    if fixed_filters:
        basic = _create_basic_filters_strategy(fixed_filters)
        if basic:
            strategies.append(basic)
    
    strategies.append(_create_no_filters_strategy())
    
    return strategies

