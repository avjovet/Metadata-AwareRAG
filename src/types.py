from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class QualityRouterOutput(BaseModel):
    """Evaluación ortográfica y corrección de preguntas."""
    has_spelling_errors: bool = Field(
        description="True si la pregunta tiene errores ortográficos, False si está correcta"
    )
    corrected_question: Optional[str] = Field(
        default=None,
        description="Pregunta corregida ortográficamente. Solo se proporciona si has_spelling_errors=True"
    )


class MainRouterOutput(BaseModel):
    """Clasificación del tipo de pregunta para enrutamiento."""
    route: Literal["simple", "compleja", "step_back"] = Field(
        description="Tipo de pregunta: 'simple' para hechos directos, 'compleja' para múltiples partes, 'step_back' para razonamiento"
    )


class QueryRewriteOutput(BaseModel):
    """Pregunta reescrita optimizada."""
    rewritten_question: str = Field(
        description="Pregunta reescrita de forma clara y concisa"
    )


class StepBackOutput(BaseModel):
    """Pregunta step-back generada."""
    step_back_question: str = Field(
        description="Pregunta más general que explore principios fundamentales"
    )


class SubQuestionsOutput(BaseModel):
    """Lista de sub-preguntas para descomposición."""
    sub_questions: List[str] = Field(
        description="Lista de sub-preguntas específicas y claras",
        min_items=1,
        max_items=5
    )


class RetrievalResult(BaseModel):
    """Resultado de la recuperación de documentos."""
    documents: List[Document] = Field(description="Documentos recuperados")
    query: str = Field(description="Consulta original")
    retrieval_method: str = Field(description="Método usado (direct, hyde, etc.)")


class RerankResult(BaseModel):
    """Resultado del re-ranking de documentos."""
    documents: List[Document] = Field(description="Documentos re-rankeados")
    original_count: int = Field(description="Número original de documentos")
    final_count: int = Field(description="Número final de documentos")


class PipelineInput(BaseModel):
    """Entrada estándar para todos los pipelines."""
    question: str = Field(description="Pregunta del usuario")


class PipelineOutput(BaseModel):
    """Salida estándar para todos los pipelines."""
    question: str = Field(description="Pregunta original")
    generated_answer: str = Field(description="Respuesta generada")
    retrieved_context: List[str] = Field(description="Contexto recuperado")
    sub_questions: Optional[List[str]] = Field(default=None, description="Sub-preguntas generadas")
    step_back_question: Optional[str] = Field(default=None, description="Pregunta step-back")
    route_quality: Optional[str] = Field(default=None, description="Clasificación de calidad")
    route: Optional[str] = Field(default=None, description="Ruta tomada en el pipeline")
    corrected_question: Optional[str] = Field(default=None, description="Pregunta corregida ortográficamente")
    error: Optional[str] = Field(default=None, description="Mensaje de error si ocurre algún problema")


class SelfQueryOutput(BaseModel):
    """Salida del Self-Query Router con metadatos de filtro."""
    query: str = Field(
        description="Consulta semántica para búsqueda vectorial"
    )
    filter: str = Field(
        description="Filtro de metadatos en formato ChromaDB (ej: 'and(eq(\"document_type\", \"laboral\"), gte(\"year\", 2020))')"
    )
    limit: Optional[int] = Field(
        default=None,
        description="Número máximo de documentos a recuperar"
    )
    reasoning: str = Field(
        description="Explicación de cómo se interpretó la pregunta para generar el filtro"
    )


class SelfQueryResult(BaseModel):
    """Resultado de la recuperación con Self-Query."""
    documents: List[Document] = Field(description="Documentos recuperados con filtros")
    query: str = Field(description="Consulta semántica usada")
    filter_applied: str = Field(description="Filtro de metadatos aplicado")
    documents_found: int = Field(description="Número de documentos encontrados")



class SemanticRouterOutput(BaseModel):
    """Clasificación semántica del documento objetivo."""
    category: Literal["constitucion", "derecho_laboral", "faq", "general"] = Field(
        description="Categoría del documento más probable basada en la semántica de la pregunta"
    )
    confidence: float = Field(
        description="Nivel de confianza en la clasificación (0.0 a 1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Explicación de por qué se clasificó en esa categoría"
    )


class ExtractedFilters(BaseModel):
    """Filtros estructurados extraídos de la pregunta."""
    article_number: Optional[int] = Field(
        default=None,
        description="Número de artículo específico mencionado"
    )
    title: Optional[str] = Field(
        default=None,
        description="Título específico de ley, decreto o documento"
    )
    year: Optional[int] = Field(
        default=None,
        description="Año específico mencionado"
    )
    source: Optional[str] = Field(
        default=None,
        description="Documento fuente específico mencionado"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Tipo de documento específico (ley, decreto, reglamento, etc.)"
    )
    topic: Optional[str] = Field(
        default=None,
        description="Tema específico (derechos fundamentales, trabajo, etc.)"
    )


class StructuredRetrievalInput(BaseModel):
    """Input estructurado para el ensamblador de recuperación."""
    semantic_query: str = Field(
        description="Consulta semántica optimizada para búsqueda vectorial"
    )
    category: str = Field(
        description="Categoría del documento objetivo"
    )
    filters: ExtractedFilters = Field(
        description="Filtros estructurados extraídos"
    )
    top_k: int = Field(
        default=15,
        description="Número máximo de documentos a recuperar"
    )