from typing import Dict, Any, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

from ..types import SemanticRouterOutput, ExtractedFilters, StructuredRetrievalInput
from ..io.llm import get_llm


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


def validate_and_normalize_filters(filters: ExtractedFilters) -> Tuple[Dict[str, Any], List[str]]:
    filter_dict = filters.dict() if hasattr(filters, 'dict') else filters
    validated_filters = {}
    discarded_filters = []
    
    for field in FILTER_PRIORITIES["redundant"]:
        if field in filter_dict and filter_dict[field] is not None:
            discarded_filters.append(f"{field}: {filter_dict[field]} (redundante)")
    
    for field in FILTER_PRIORITIES["primary"]:
        if field in filter_dict and filter_dict[field] is not None:
            value = filter_dict[field]
            
            if field in VALID_VALUES and value not in VALID_VALUES[field]:
                discarded_filters.append(f"{field}: {value} (valor inválido)")
            else:
                validated_filters[field] = value
    
    for field in FILTER_PRIORITIES["secondary"]:
        if field in filter_dict and filter_dict[field] is not None:
            value = filter_dict[field]
            
            if field == "title":
                if value in GENERIC_VALUES.get("title", []):
                    discarded_filters.append(f"{field}: {value} (genérico)")
                    continue
                
                if "document_type" in validated_filters:
                    doc_type = validated_filters["document_type"]
                    if doc_type in VALID_VALUES["title"] and value not in VALID_VALUES["title"][doc_type]:
                        discarded_filters.append(f"{field}: {value} (no coincide con {doc_type})")
                        continue
                
                validated_filters[field] = value
                
            elif field == "article_number":
                if isinstance(value, int) and 1 <= value <= 206:
                    validated_filters[field] = value
                else:
                    discarded_filters.append(f"{field}: {value} (número inválido)")
                    
            elif field == "year":
                if isinstance(value, int) and 1990 <= value <= 2024:
                    validated_filters[field] = value
                else:
                    discarded_filters.append(f"{field}: {value} (año inválido)")
    
    return validated_filters, discarded_filters


def create_filter_strategies(validated_filters: Dict[str, Any], semantic_category: str) -> List[Dict[str, Any]]:
    strategies = []
    
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
    
    fixed_filters = fixed_metadata.get(semantic_category, {})
    
    has_variables = any(value is not None for value in validated_filters.values())
    
    if has_variables and fixed_filters:
        all_filters = fixed_filters.copy()
        for key, value in validated_filters.items():
            if value is not None:
                all_filters[key] = value
        
        strategies.append({
            "name": "todos_filtros",
            "filters": all_filters,
            "description": f"Todos los filtros: {list(all_filters.keys())}"
        })
        
        filters_without_title = fixed_filters.copy()
        for key, value in validated_filters.items():
            if value is not None and key != 'title':
                filters_without_title[key] = value
        
        if len(filters_without_title) > len(fixed_filters):
            strategies.append({
                "name": "sin_title",
                "filters": filters_without_title,
                "description": f"Sin title: {list(filters_without_title.keys())}"
            })
        
        filters_without_doc_type = fixed_filters.copy()
        if 'document_type' in filters_without_doc_type:
            del filters_without_doc_type['document_type']
        for key, value in validated_filters.items():
            if value is not None and key not in ['title', 'document_type']:
                filters_without_doc_type[key] = value
        
        if len(filters_without_doc_type) > 0:
            strategies.append({
                "name": "sin_document_type",
                "filters": filters_without_doc_type,
                "description": f"Sin document_type: {list(filters_without_doc_type.keys())}"
            })
        
        filters_without_year = fixed_filters.copy()
        for key in ['document_type', 'year']:
            if key in filters_without_year:
                del filters_without_year[key]
        for key, value in validated_filters.items():
            if value is not None and key not in ['title', 'document_type', 'year']:
                filters_without_year[key] = value
        
        if len(filters_without_year) > 0:
            strategies.append({
                "name": "sin_year",
                "filters": filters_without_year,
                "description": f"Sin year: {list(filters_without_year.keys())}"
            })
    
    if fixed_filters:
        basic_filters = {}
        for key in ['source', 'topic']:
            if key in fixed_filters:
                basic_filters[key] = fixed_filters[key]
        
        if basic_filters:
            strategies.append({
                "name": "solo_basicos",
                "filters": basic_filters,
                "description": f"Solo básicos: {list(basic_filters.keys())}"
            })
    
    strategies.append({
        "name": "sin_filtros",
        "filters": {},
        "description": "Búsqueda semántica pura en toda la BD"
    })
    
    return strategies


def clean_semantic_response(json_data: dict) -> dict:
    if isinstance(json_data, str):
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_data, re.DOTALL)
        if json_match:
            try:
                import json
                json_data = json.loads(json_match.group(1))
            except:
                try:
                    json_data = json.loads(json_data)
                except:
                    return {
                        'category': 'general',
                        'confidence': 0.5,
                        'reasoning': 'Error en parsing JSON'
                    }
    
    valid_fields = {'category', 'confidence', 'reasoning'}
    
    cleaned_data = {}
    
    for field, value in json_data.items():
        if field in valid_fields:
            if field == 'confidence' and value is not None:
                try:
                    cleaned_data[field] = float(value)
                except (ValueError, TypeError):
                    cleaned_data[field] = 0.5
            else:
                cleaned_data[field] = value
    
    for field in valid_fields:
        if field not in cleaned_data:
            if field == 'confidence':
                cleaned_data[field] = 0.5
            elif field == 'category':
                cleaned_data[field] = 'general'
            else:
                cleaned_data[field] = 'Sin razonamiento'
    
    return cleaned_data


def create_semantic_router(llm) -> RunnableLambda:
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
    
    def debug_semantic_router(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("question", "")
        
        try:
            response = llm.invoke(semantic_router_prompt.format(question=question))
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            import json
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
            
        except Exception:
            return {
                **inputs,
                "semantic_category": "general",
                "semantic_confidence": 0.5,
                "semantic_reasoning": "Error en clasificación"
            }
    
    return RunnableLambda(debug_semantic_router)


def create_filter_extractor(llm) -> RunnableLambda:
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
    
    def debug_filter_extractor(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("question", "")
        
        try:
            response = llm.invoke(filter_extractor_prompt.format(question=question))
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            import json
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
            
        except Exception:
            import re
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
    
    return RunnableLambda(debug_filter_extractor)


def clean_json_response(json_data: dict) -> dict:
    if isinstance(json_data, str):
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_data, re.DOTALL)
        if json_match:
            try:
                import json
                json_data = json.loads(json_match.group(1))
            except:
                try:
                    json_data = json.loads(json_data)
                except:
                    return {
                        'article_number': None,
                        'title': None,
                        'year': None,
                        'source': None,
                        'document_type': None,
                        'topic': None
                    }
    
    valid_fields = {
        'article_number', 'title', 'year', 'source', 'document_type', 'topic'
    }
    
    cleaned_data = {}
    
    for field, value in json_data.items():
        if field in valid_fields:
            if field == 'article_number' and value is not None:
                try:
                    cleaned_data[field] = int(value)
                except (ValueError, TypeError):
                    cleaned_data[field] = None
            elif field == 'year' and value is not None:
                try:
                    cleaned_data[field] = int(value)
                except (ValueError, TypeError):
                    cleaned_data[field] = None
            else:
                cleaned_data[field] = value
    
    for field in valid_fields:
        if field not in cleaned_data:
            cleaned_data[field] = None
    
    return cleaned_data


def build_chromadb_filter(filters: Dict[str, Any]) -> Optional[dict]:
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


def create_retrieval_assembler(vectorstore, top_k: int = 15) -> RunnableLambda:
    def debug_retrieval_assembler(inputs: Dict[str, Any]) -> List[Document]:
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
            
        except Exception:
            try:
                basic_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
                docs = basic_retriever.invoke(question)
                return docs
            except:
                return []
    
    return RunnableLambda(debug_retrieval_assembler)


def create_modular_self_query_pipeline(llm, vectorstore, top_k: int = 15):
    semantic_router = create_semantic_router(llm)
    filter_extractor = create_filter_extractor(llm)
    retrieval_assembler = create_retrieval_assembler(vectorstore, top_k)
    
    return {
        "semantic_router": semantic_router,
        "filter_extractor": filter_extractor,
        "retrieval_assembler": retrieval_assembler
    }


def create_self_query_retriever(llm, vectorstore, top_k: int = 15):
    def simple_self_query_retriever(question: str) -> List[Document]:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(question)
            return docs
        except Exception:
            return []
    
    return simple_self_query_retriever
