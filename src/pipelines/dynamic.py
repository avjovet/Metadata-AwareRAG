import logging
from typing import Dict, Any, List, Optional
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableLambda, 
    RunnableBranch, 
    RunnableParallel,
    Runnable
)
from langchain_core.documents import Document

from ..io.vectordb import get_vector_store, get_self_query_retriever
from ..io.llm import get_llm
from ..steps.retrieval import create_retrieval_chain
from ..steps.rerank import create_reranker, rerank_documents
from ..steps.routing import (
    create_quality_router,
    create_main_router,
    create_decomposition_chain,
    create_step_back_generator
)
from ..steps.self_query import create_self_query_retriever
from ..steps.self_query import create_modular_self_query_pipeline
from ..steps.synthesis import (
    create_rag_answer_chain,
    create_complex_branch_chain,
    create_step_back_branch_chain
)
from ..types import PipelineOutput, SemanticRouterOutput, ExtractedFilters
from ..config.settings import settings
from ..utils.validators import QuestionValidator

logger = logging.getLogger(__name__)


def create_dynamic_rag_pipeline(
    db_folder_name: str,
    embedding_model_name: str,
    llm_model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    enable_self_query: Optional[bool] = None
) -> Runnable:
    """
    Crea un pipeline RAG dinámico con routing, filtrado adaptativo y reranking.
    
    Args:
        db_folder_name: Nombre de la carpeta de la base de datos vectorial
        embedding_model_name: Nombre del modelo de embeddings
        llm_model_name: Nombre del modelo LLM (por defecto desde settings)
        temperature: Temperatura para el LLM (por defecto 0.0)
        top_k: Número de documentos a recuperar (por defecto 15)
        enable_self_query: Si habilitar self-query con filtros de metadatos
        
    Returns:
        Runnable que implementa el pipeline RAG dinámico completo
    """
    llm_model_name = llm_model_name or settings.OLLAMA_MODEL
    temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
    top_k = top_k or settings.DYNAMIC_PIPELINE_TOP_K
    enable_self_query = enable_self_query if enable_self_query is not None else True
    
    vector_store = get_vector_store(db_folder_name, embedding_model_name)
    llm = get_llm(model_name=llm_model_name, temperature=temperature)
    reranker = create_reranker()
    
    # Crear quality_router una vez (optimización de performance)
    quality_router = create_quality_router(llm)
    
    # Crear retriever básico una vez para fallback (optimización)
    basic_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    # Crear componentes de self-query solo si está habilitado
    if enable_self_query:
        modular_components = create_modular_self_query_pipeline(llm, vector_store, top_k=top_k)
        self_query_retriever = create_self_query_retriever(llm, vector_store, top_k=top_k)
    else:
        modular_components = None
        self_query_retriever = None
    
    def quality_router_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("question", "NO_QUESTION")
        
        try:
            quality_result = quality_router.invoke({"question": question})
            
            if isinstance(quality_result, dict):
                merged_result = {**inputs, **quality_result}
                return merged_result
            else:
                # Fallback seguro: establecer original_question manualmente
                logger.warning(f"Quality router retornó tipo inesperado: {type(quality_result)}")
                return {
                    **inputs,
                    "original_question": inputs.get("question", ""),
                    "has_spelling_errors": False
                }
                
        except Exception as e:
            logger.error(f"Error en quality router: {e}", exc_info=True)
            return {
                **inputs,
                "original_question": inputs.get("question", ""),
                "has_spelling_errors": False,
                "corrected_question": None,
                "correction_notes": f"Error en corrección: {e}"
            }

    quality_router_chain = RunnableLambda(quality_router_step)

    def rerank_docs(inputs: Dict[str, Any]) -> List[Document]:
        question = inputs.get("question", "NO_QUESTION")
        
        if "retrieved_docs" not in inputs:
            return []
        
        docs = inputs["retrieved_docs"]
        
        if not docs:
            return []
        
        try:
            rerank_result = rerank_documents(question, docs, reranker)
            return rerank_result.documents
        except Exception as e:
            logger.warning(f"Error en reranking, usando documentos originales: {e}")
            return docs

    rerank_chain = RunnableLambda(rerank_docs)

    rag_answer_chain = create_rag_answer_chain(llm)
    
    def modular_self_query_step(inputs: Dict[str, Any]) -> List[Document]:
        """
        Ejecuta el pipeline modular de self-query con fallbacks progresivos.
        
        Estrategia de fallback:
        1. Pipeline modular completo (semantic router + filter extractor + retrieval assembler)
        2. Self-query retriever simple
        3. Retriever básico sin filtros
        4. Retornar lista vacía si todo falla
        """
        question = inputs.get("question", "NO_QUESTION")
        
        # Si self-query está deshabilitado, ir directamente al retriever básico
        if not enable_self_query:
            try:
                docs = basic_retriever.invoke(question)
                logger.debug(f"Retriever básico (self-query deshabilitado) para pregunta '{question}': {len(docs)} documentos recuperados")
                return docs
            except Exception as basic_error:
                logger.error(f"Error en retriever básico para pregunta '{question}': {basic_error}", exc_info=True)
                return []
        
        # Intento 1: Pipeline modular completo (enable_self_query=True implica modular_components is not None)
        else:
            try:
                # Paso 1: Clasificación semántica
                try:
                    semantic_result = modular_components["semantic_router"].invoke(inputs)
                except Exception as semantic_error:
                    logger.warning(
                        f"Error en semantic router para pregunta '{question}': {semantic_error}. "
                        "Usando categoría por defecto 'general'."
                    )
                    semantic_result = {
                        **inputs,
                        "semantic_category": "general",
                        "semantic_confidence": 0.5,
                        "semantic_reasoning": f"Error en clasificación: {semantic_error}"
                    }
                
                # Paso 2: Extracción de filtros
                try:
                    filter_result = modular_components["filter_extractor"].invoke(semantic_result)
                except Exception as filter_error:
                    logger.warning(
                        f"Error en filter extractor para pregunta '{question}': {filter_error}. "
                        "Usando filtros vacíos."
                    )
                    filter_result = {
                        **semantic_result,
                        "extracted_filters": ExtractedFilters()
                    }
                
                # Paso 3: Recuperación con filtros
                try:
                    docs = modular_components["retrieval_assembler"].invoke(filter_result)
                    logger.debug(f"Pipeline modular exitoso para pregunta '{question}': {len(docs)} documentos recuperados")
                    return docs
                except Exception as retrieval_error:
                    logger.warning(
                        f"Error en retrieval assembler para pregunta '{question}': {retrieval_error}. "
                        "Intentando fallback a self-query retriever."
                    )
                    # No hacer raise aquí, continuar con fallback
            
            except Exception as modular_error:
                logger.warning(
                    f"Error crítico en pipeline modular para pregunta '{question}': {modular_error}. "
                    "Intentando fallback a self-query retriever."
                )
        
        # Intento 2: Self-query retriever simple
        if self_query_retriever is not None:
            try:
                docs = self_query_retriever.invoke(question)
                logger.debug(f"Self-query retriever exitoso para pregunta '{question}': {len(docs)} documentos recuperados")
                return docs
            except Exception as self_query_error:
                logger.warning(
                    f"Error en self-query retriever para pregunta '{question}': {self_query_error}. "
                    "Intentando fallback a retriever básico."
                )
        
        # Intento 3: Retriever básico sin filtros (ya creado arriba)
        try:
            docs = basic_retriever.invoke(question)
            logger.debug(f"Retriever básico exitoso para pregunta '{question}': {len(docs)} documentos recuperados")
            return docs
        except Exception as basic_error:
            logger.error(
                f"Error en retriever básico para pregunta '{question}': {basic_error}. "
                "Todos los métodos de recuperación fallaron. Retornando lista vacía.",
                exc_info=True
            )
            return []
    
    self_query_chain = RunnableLambda(modular_self_query_step)
    
    final_chain = (
        quality_router_chain  # quality_router ya establece original_question
        | RunnablePassthrough.assign(retrieved_docs=self_query_chain)
        # Nota: La sobrescritura de retrieved_docs es intencional - rerank_chain reemplaza
        # los documentos recuperados con la versión rerankeada para mejorar relevancia
        | RunnablePassthrough.assign(retrieved_docs=rerank_chain)
        | RunnablePassthrough.assign(
            generated_answer=RunnableLambda(lambda x: {
                "context": "\n\n".join([d.page_content for d in x.get("retrieved_docs", [])]) if x.get("retrieved_docs", []) else "",
                "question": x.get("question", "")
            })
            | rag_answer_chain
        )
    )

    def format_output(chain_result: Dict) -> PipelineOutput:
        retrieved_docs = chain_result.get("retrieved_docs", [])
        
        # Validación defensiva: asegurar que original_question existe
        original_question = chain_result.get("original_question") or chain_result.get("question", "")
        
        route_quality_value = None
        if "has_spelling_errors" in chain_result:
            has_errors = chain_result.get("has_spelling_errors", False)
            route_quality_value = "mal_redactada" if has_errors else "simple"

        output = PipelineOutput(
            question=original_question,
            generated_answer=chain_result.get("generated_answer", ""),
            retrieved_context=[doc.page_content for doc in retrieved_docs],
            route_quality=route_quality_value,
            route="simplified",
            corrected_question=chain_result.get("question") if chain_result.get("has_spelling_errors") else None
        )
        
        return output
        
    return final_chain | RunnableLambda(format_output)


def invoke_dynamic_pipeline(chain: Runnable, question: str) -> PipelineOutput:
    """
    Ejecuta el pipeline dinámico con una pregunta.
    
    Args:
        chain: Cadena del pipeline dinámico
        question: Pregunta del usuario
        
    Returns:
        PipelineOutput con respuesta generada, contexto recuperado y metadatos del proceso
    """
    is_valid, error_message = QuestionValidator.validate(question)
    if not is_valid:
        return PipelineOutput(
            question=question or "",
            generated_answer="",
            retrieved_context=[],
            error=error_message
        )
    
    try:
        result = chain.invoke({"question": question})
        return result
    except Exception as e:
        logger.error(f"Error en dynamic pipeline: {e}", exc_info=True)
        return PipelineOutput(
            question=question,
            generated_answer=f"Error en el pipeline: {str(e)}",
            retrieved_context=[],
            error=str(e)
        )
