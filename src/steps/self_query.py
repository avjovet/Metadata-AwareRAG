import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

from ..types import SemanticRouterOutput, ExtractedFilters, StructuredRetrievalInput
from ..io.llm import get_llm
from ..config.constants import LegalConstants


FILTER_PRIORITIES = {
    "primary": ["document_type", "source"],
    "secondary": ["article_number", "year", "title"],
    "redundant": ["topic"]
}

VALID_VALUES = {
    "document_type": ["constitucion", "decreto", "faq"],
    "source": ["Constitución Política del Perú", "Compendio Derecho Laboral", "Preguntas Frecuentes"],
    "title": {
        "constitucion": [f"Artículo {i}" for i in range(1, 207)],
        "decreto": ["Decreto Legislativo N.° 728", "Decreto Legislativo N.° 713", "Decreto Legislativo N.° 650"],
        "faq": [f"Pregunta Frecuente {i}" for i in range(1, 11)]
    }
}

GENERIC_VALUES = {
    "title": ["Constitución", "Constitución Política", "Decreto", "Decreto Legislativo", "FAQ", "Pregunta"],
    "source": ["Constitución", "Ley", "Decreto", "FAQ"]
}


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
) -> Dict[str, Any] | None:
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
) -> Dict[str, Any] | None:
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
) -> Dict[str, Any] | None:
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


def _create_basic_filters_strategy(fixed_filters: Dict[str, Any]) -> Dict[str, Any] | None:
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
    try:
        structured_llm = llm.with_structured_output(SemanticRouterOutput)
    except Exception:
        structured_llm = llm
    
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


def create_filter_extractor(llm: BaseLanguageModel) -> Runnable:
    """
    Crea un extractor de filtros para extraer metadatos variables de preguntas.
    
    Args:
        llm: Modelo de lenguaje para extracción de metadatos
        
    Returns:
        Runnable que extrae filtros como article_number, title, year de la pregunta
    """
    try:
        structured_llm = llm.with_structured_output(ExtractedFilters)
    except Exception:
        structured_llm = llm
    
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
            
            for i, strategy in enumerate(strategies, 1):
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
        except Exception:
            return []
    
    return simple_self_query_retriever
