"""Utilidades para parsing y limpieza de respuestas JSON del LLM."""

import json
import re
from typing import Any, Dict, Set


def parse_llm_json_response(
    response: str | dict,
    expected_fields: Set[str],
    default_values: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Parsea y limpia respuestas JSON del LLM de forma genérica.
    
    Args:
        response: Respuesta del LLM (puede ser string o dict)
        expected_fields: Conjunto de campos esperados en la respuesta
        default_values: Valores por defecto para campos faltantes o inválidos
        
    Returns:
        Diccionario limpio con campos válidos y valores por defecto para campos faltantes
    """
    # Si es string, intentar extraer JSON de markdown code blocks
    if isinstance(response, str):
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                response = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                try:
                    response = json.loads(response.strip())
                except json.JSONDecodeError:
                    return default_values.copy()
        else:
            try:
                response = json.loads(response.strip())
            except json.JSONDecodeError:
                return default_values.copy()
    
    # Validar y limpiar campos
    cleaned_data = {}
    
    for field in expected_fields:
        if field in response:
            value = response[field]
            # Validación específica por tipo de campo
            if field == 'confidence' and value is not None:
                try:
                    cleaned_data[field] = float(value)
                except (ValueError, TypeError):
                    cleaned_data[field] = default_values.get(field, 0.5)
            elif field in ['article_number', 'year'] and value is not None:
                try:
                    cleaned_data[field] = int(value)
                except (ValueError, TypeError):
                    cleaned_data[field] = default_values.get(field)
            else:
                cleaned_data[field] = value
        else:
            cleaned_data[field] = default_values.get(field)
    
    return cleaned_data

