"""Validación y normalización de filtros extraídos de preguntas."""

from typing import Any, Dict, List, Tuple
from ..types import ExtractedFilters
from ..config.constants import LegalConstants
from .constants import FILTER_PRIORITIES, VALID_VALUES, GENERIC_VALUES


def _validate_title_field(
    value: Any, 
    validated_filters: Dict[str, Any], 
    discarded_filters: List[str]
) -> bool:
    """
    Valida el campo title.
    
    Args:
        value: Valor del campo title
        validated_filters: Diccionario de filtros validados (se actualiza si es válido)
        discarded_filters: Lista de filtros descartados (se actualiza si es inválido)
        
    Returns:
        True si el campo es válido y se agregó a validated_filters, False en caso contrario
    """
    if value in GENERIC_VALUES.get("title", []):
        discarded_filters.append(f"title: {value} (genérico)")
        return False
    
    if "document_type" in validated_filters:
        doc_type = validated_filters["document_type"]
        if doc_type in VALID_VALUES["title"] and value not in VALID_VALUES["title"][doc_type]:
            discarded_filters.append(f"title: {value} (no coincide con {doc_type})")
            return False
    
    validated_filters["title"] = value
    return True


def _validate_article_number_field(
    value: Any, 
    validated_filters: Dict[str, Any], 
    discarded_filters: List[str]
) -> bool:
    """
    Valida el campo article_number.
    
    Args:
        value: Valor del campo article_number
        validated_filters: Diccionario de filtros validados (se actualiza si es válido)
        discarded_filters: Lista de filtros descartados (se actualiza si es inválido)
        
    Returns:
        True si el campo es válido y se agregó a validated_filters, False en caso contrario
    """
    if isinstance(value, int) and LegalConstants.MIN_ARTICLE_NUMBER <= value <= LegalConstants.MAX_ARTICLE_NUMBER:
        validated_filters["article_number"] = value
        return True
    else:
        discarded_filters.append(f"article_number: {value} (número inválido)")
        return False


def _validate_year_field(
    value: Any, 
    validated_filters: Dict[str, Any], 
    discarded_filters: List[str]
) -> bool:
    """
    Valida el campo year.
    
    Args:
        value: Valor del campo year
        validated_filters: Diccionario de filtros validados (se actualiza si es válido)
        discarded_filters: Lista de filtros descartados (se actualiza si es inválido)
        
    Returns:
        True si el campo es válido y se agregó a validated_filters, False en caso contrario
    """
    if isinstance(value, int) and LegalConstants.MIN_YEAR <= value <= LegalConstants.MAX_YEAR:
        validated_filters["year"] = value
        return True
    else:
        discarded_filters.append(f"year: {value} (año inválido)")
        return False


def validate_and_normalize_filters(filters: ExtractedFilters) -> Tuple[Dict[str, Any], List[str]]:
    """
    Valida y normaliza filtros extraídos de una pregunta.
    
    Los filtros se validan según prioridades:
    1. Filtros redundantes se descartan automáticamente
    2. Filtros primarios se validan contra valores permitidos
    3. Filtros secundarios tienen validación específica por tipo
    
    Args:
        filters: Filtros extraídos del LLM
        
    Returns:
        Tuple de (filtros_validados, filtros_descartados)
        - filtros_validados: Diccionario con filtros que pasaron la validación
        - filtros_descartados: Lista de strings describiendo filtros descartados y razón
    """
    filter_dict = filters.dict() if hasattr(filters, 'dict') else filters
    validated_filters = {}
    discarded_filters = []
    
    # Descartar filtros redundantes
    for field in FILTER_PRIORITIES["redundant"]:
        if field in filter_dict and filter_dict[field] is not None:
            discarded_filters.append(f"{field}: {filter_dict[field]} (redundante)")
    
    # Validar filtros primarios
    for field in FILTER_PRIORITIES["primary"]:
        if field in filter_dict and filter_dict[field] is not None:
            value = filter_dict[field]
            
            if field in VALID_VALUES and value not in VALID_VALUES[field]:
                discarded_filters.append(f"{field}: {value} (valor inválido)")
            else:
                validated_filters[field] = value
    
    # Validar filtros secundarios con lógica específica
    for field in FILTER_PRIORITIES["secondary"]:
        if field in filter_dict and filter_dict[field] is not None:
            value = filter_dict[field]
            
            if field == "title":
                _validate_title_field(value, validated_filters, discarded_filters)
            elif field == "article_number":
                _validate_article_number_field(value, validated_filters, discarded_filters)
            elif field == "year":
                _validate_year_field(value, validated_filters, discarded_filters)
    
    return validated_filters, discarded_filters

