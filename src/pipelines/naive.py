from typing import Dict, Any, Optional
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser

from ..io.vectordb import get_vector_store
from ..io.llm import get_llm
from ..steps.prompts import RAG_BASIC_PROMPT
from ..steps.retrieval import docs_to_text
from ..types import PipelineInput, PipelineOutput
from ..config.settings import settings
from ..utils.validators import QuestionValidator


def create_naive_rag_pipeline(
    db_folder_name: str,
    embedding_model_name: str,
    llm_model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None
) -> Runnable:
    """
    Crea un pipeline RAG naive (básico) sin routing ni filtrado avanzado.
    
    Args:
        db_folder_name: Nombre de la carpeta de la base de datos vectorial
        embedding_model_name: Nombre del modelo de embeddings
        llm_model_name: Nombre del modelo LLM (por defecto desde settings)
        temperature: Temperatura para el LLM (por defecto 0.1)
        top_k: Número de documentos a recuperar (por defecto 5)
        
    Returns:
        Runnable que implementa el pipeline RAG naive
    """
    llm_model_name = llm_model_name or settings.OLLAMA_MODEL
    temperature = temperature if temperature is not None else settings.NAIVE_PIPELINE_TEMPERATURE
    top_k = top_k or settings.NAIVE_PIPELINE_TOP_K
    
    vector_store = get_vector_store(db_folder_name, embedding_model_name)
    llm = get_llm(model_name=llm_model_name, temperature=temperature)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    rag_chain_from_docs = (RAG_BASIC_PROMPT | llm | StrOutputParser())
    
    chain = (
        RunnablePassthrough.assign(original_docs=RunnableLambda(lambda x: x["question"]) | retriever)
        .assign(context=lambda x: docs_to_text(x["original_docs"]))
        .assign(answer=rag_chain_from_docs)
    )
    
    return chain


def invoke_naive_pipeline(chain: Runnable, question: str) -> PipelineOutput:
    """
    Ejecuta el pipeline naive con una pregunta.
    
    Args:
        chain: Cadena del pipeline naive
        question: Pregunta del usuario
        
    Returns:
        PipelineOutput con respuesta generada y contexto recuperado
    """
    is_valid, error_message = QuestionValidator.validate(question)
    if not is_valid:
        return PipelineOutput(
            question=question or "",
            generated_answer="",
            retrieved_context=[],
            error=error_message
        )
    
    output = chain.invoke({"question": question})
    
    return PipelineOutput(
        question=question,
        generated_answer=output["answer"],
        retrieved_context=[doc.page_content for doc in output["original_docs"]]
    )
