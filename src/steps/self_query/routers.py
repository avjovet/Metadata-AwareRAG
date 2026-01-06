"""Routers semánticos y extractores de filtros para self-query."""

import json
import re
import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from ..types import SemanticRouterOutput, ExtractedFilters
from ..utils.json_utils import parse_llm_json_response

logger = logging.getLogger(__name__)


def clean_semantic_response(json_data: dict) -> dict:
    """
    Limpia y valida respuesta JSON del router semántico.
    
    Args:
        json_data: Datos JSON (puede ser string o dict)
        
    Returns:
        Diccionario limpio con campos válidos: category, confidence, reasoning
    """
    return parse_llm_json_response(
        response=json_data,
        expected_fields={'category', 'confidence', 'reasoning'},
        default_values={
            'category': 'general',
            'confidence': 0.5,
            'reasoning': 'Sin razonamiento'
        }
    )


def create_semantic_router(llm: BaseLanguageModel) -> Runnable:
    """
    Crea un router semántico para clasificar el tipo de documento objetivo.
    
    Args:
        llm: Modelo de lenguaje para clasificación semántica
        
    Returns:
        Runnable que clasifica preguntas en categorías: constitucion, derecho_laboral, faq, general
    """
    # Intentar crear structured_llm, usar booleano para tracking
    use_structured = False
    structured_llm = None
    try:
        structured_llm = llm.with_structured_output(SemanticRouterOutput)
        use_structured = True
    except (AttributeError, TypeError) as e:
        logger.debug(f"LLM no soporta structured_output, usando LLM estándar: {e}")
    
    semantic_router_prompt = PromptTemplate.from_template("""
Eres un experto en clasificación de documentos legales peruanos. Tu tarea es determinar qué tipo de documento es más probable que contenga la respuesta a la pregunta del usuario.

CATEGORÍAS DISPONIBLES:
- constitucion: Preguntas sobre la Constitución Política del Perú, derechos fundamentales, organización del Estado, poderes públicos
- derecho_laboral: Preguntas sobre relaciones laborales, contratos de trabajo, derechos de trabajadores, despidos, beneficios sociales
- faq: Preguntas frecuentes generales, procedimientos comunes, dudas básicas sobre trámites

EJEMPLOS:
- "¿Qué dice el artículo 2 de la Constitución?" → constitucion (confianza: 0.95)
- "¿Cuáles son los derechos fundamentales?" → constitucion (confianza: 0.90)
- "¿Cómo funciona el despido arbitrario?" → derecho_laboral (confianza: 0.95)
- "¿Qué beneficios sociales tiene un trabajador?" → derecho_laboral (confianza: 0.90)
- "¿Cómo renovar mi DNI?" → faq (confianza: 0.85)


CRITERIOS DE CONFIANZA:
- 0.9-1.0: Muy específico de la categoría, términos técnicos claros
- 0.7-0.9: Claramente relacionado pero menos específico
- 0.5-0.7: Posiblemente relacionado, algunos indicadores
- 0.0-0.5: Incierto o requiere múltiples categorías

IMPORTANTE: Responde SOLO con el JSON válido en el formato exacto requerido.

Pregunta: {question}

Clasifica esta pregunta determinando:
1. La categoría más probable
2. Tu nivel de confianza (0.0 a 1.0)
3. El razonamiento detrás de tu decisión
""")
    
    def semantic_router_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("question", "")
        
        try:
            # Intentar usar structured_llm si está disponible
            if use_structured and structured_llm is not None:
                try:
                    # Usar formato de mensajes para structured_llm (como en routing.py)
                    result = structured_llm.invoke([
                        ("system", "Eres un experto en clasificación de documentos legales peruanos. Tu tarea es determinar qué tipo de documento es más probable que contenga la respuesta a la pregunta del usuario."),
                        ("human", f"Clasifica esta pregunta determinando la categoría, confianza y razonamiento: '{question}'")
                    ])
                    # structured_llm devuelve directamente el objeto Pydantic
                    if isinstance(result, SemanticRouterOutput):
                        return {
                            **inputs,
                            "semantic_category": result.category,
                            "semantic_confidence": result.confidence,
                            "semantic_reasoning": result.reasoning
                        }
                except Exception as structured_error:
                    logger.debug(f"Error usando structured_output, fallback a parsing manual: {structured_error}")
            
            # Fallback: usar llm estándar y parsear respuesta
            response = llm.invoke(semantic_router_prompt.format(question=question))
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            try:
                json_data = json.loads(response_text.strip())
            except json.JSONDecodeError:
                json_data = response_text.strip()
            
            cleaned_json_data = clean_semantic_response(json_data)

            category = cleaned_json_data.get('category', 'general')
            confidence = cleaned_json_data.get('confidence', 0.5)
            reasoning = cleaned_json_data.get('reasoning', 'JSON parsing')
            
            return {
                **inputs,
                "semantic_category": category,
                "semantic_confidence": confidence,
                "semantic_reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Error en semantic router: {e}", exc_info=True)
            return {
                **inputs,
                "semantic_category": "general",
                "semantic_confidence": 0.5,
                "semantic_reasoning": "Error en clasificación"
            }
    
    return RunnableLambda(semantic_router_step)


def clean_json_response(json_data: dict) -> dict:
    """
    Limpia y valida respuesta JSON del extractor de filtros.
    
    Args:
        json_data: Datos JSON (puede ser string o dict)
        
    Returns:
        Diccionario limpio con campos válidos de filtros extraídos
    """
    return parse_llm_json_response(
        response=json_data,
        expected_fields={'article_number', 'title', 'year', 'source', 'document_type', 'topic'},
        default_values={
            'article_number': None,
            'title': None,
            'year': None,
            'source': None,
            'document_type': None,
            'topic': None
        }
    )


def create_filter_extractor(llm: BaseLanguageModel) -> Runnable:
    """
    Crea un extractor de filtros para extraer metadatos variables de preguntas.
    
    Args:
        llm: Modelo de lenguaje para extracción de metadatos
        
    Returns:
        Runnable que extrae filtros como article_number, title, year de la pregunta
    """
    # Intentar crear structured_llm, usar booleano para tracking
    use_structured = False
    structured_llm = None
    try:
        structured_llm = llm.with_structured_output(ExtractedFilters)
        use_structured = True
    except (AttributeError, TypeError) as e:
        logger.debug(f"LLM no soporta structured_output, usando LLM estándar: {e}")
    
    filter_extractor_prompt = PromptTemplate.from_template("""
Eres un experto en análisis de texto legal. Tu tarea es extraer ÚNICAMENTE los metadatos variables mencionados explícitamente en la pregunta.

ESTRUCTURA DE DATOS REAL:
Los metadatos fijos ya están determinados por el router semántico:
- CONSTITUCIÓN: source="Constitución Política del Perú", document_type="constitucion", topic="derechos_fundamentales"
- COMPENDIO LABORAL: source="Compendio Derecho Laboral", document_type="decreto", topic="derecho_laboral"  
- PREGUNTAS FRECUENTES: source="Preguntas Frecuentes", document_type="faq", topic="Preguntas Frecuentes"

METADATOS VARIABLES A EXTRAER:
- article_number: Número de artículo (ej: "artículo 2" → 2, "artículo 139" → 139)
- title: Título específico del documento (ej: "Decreto Supremo N.º 003-97-TR", "Pregunta Frecuente - II: REGLAMENTO INTERNO DE TRABAJO")
- year: Año específico (ej: "1993", "1997")

REGLAS ESTRICTAS:
1. Solo extrae información EXPLÍCITAMENTE mencionada
2. Si no se menciona específicamente, deja el campo como null
3. NO extraigas source, document_type ni topic (ya los decide el router semántico)
4. Responde SOLO con el JSON válido en el formato exacto requerido

EJEMPLOS:
- "¿Qué dice el artículo 2 de la Constitución?"
  → article_number: 2, title: null, year: null

- "¿Qué dice el Decreto Supremo 003-97-TR?"
  → article_number: null, title: "Decreto Supremo N.º 003-97-TR", year: 1997

- "¿Hay preguntas frecuentes sobre el reglamento interno?"
  → article_number: null, title: null, year: null

- "¿Qué leyes de 1991 existen?"
  → article_number: null, title: null, year: 1991

- "¿Cómo funciona el gobierno?"
  → article_number: null, title: null, year: null

Pregunta: {question}

Extrae SOLO los metadatos variables mencionados:
""")
    
    def filter_extractor_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("question", "")
        
        try:
            # Intentar usar structured_llm si está disponible
            if use_structured and structured_llm is not None:
                try:
                    # Usar formato de mensajes para structured_llm (como en routing.py)
                    result = structured_llm.invoke([
                        ("system", "Eres un experto en análisis de texto legal. Tu tarea es extraer ÚNICAMENTE los metadatos variables mencionados explícitamente en la pregunta."),
                        ("human", f"Extrae los metadatos variables de esta pregunta: '{question}'")
                    ])
                    # structured_llm devuelve directamente el objeto Pydantic
                    if isinstance(result, ExtractedFilters):
                        return {
                            **inputs,
                            "extracted_filters": result
                        }
                except Exception as structured_error:
                    logger.debug(f"Error usando structured_output, fallback a parsing manual: {structured_error}")
            
            # Fallback: usar llm estándar y parsear respuesta
            response = llm.invoke(filter_extractor_prompt.format(question=question))
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            try:
                json_data = json.loads(response_text.strip())
            except json.JSONDecodeError:
                json_data = response_text.strip()
            
            cleaned_data = clean_json_response(json_data)
            
            filters = ExtractedFilters(
                article_number=cleaned_data.get('article_number'),
                title=cleaned_data.get('title'),
                year=cleaned_data.get('year'),
                source=cleaned_data.get('source'),
                document_type=cleaned_data.get('document_type'),
                topic=cleaned_data.get('topic')
            )
            
            return {
                **inputs,
                "extracted_filters": filters
            }
            
        except Exception as e:
            logger.error(f"Error en filter extractor: {e}", exc_info=True)
            filters = ExtractedFilters()
            
            article_match = re.search(r'artículo\s+(\d+)', question, re.IGNORECASE)
            if article_match:
                filters.article_number = int(article_match.group(1))
            
            if any(word in question.lower() for word in ['constitución', 'constitución política']):
                filters.source = "Constitución Política del Perú"
                filters.document_type = "constitucion"
            
            return {
                **inputs,
                "extracted_filters": filters
            }
    
    return RunnableLambda(filter_extractor_step)

