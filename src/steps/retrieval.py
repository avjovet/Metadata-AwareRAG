import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel

from ..types import RetrievalResult
from ..steps.prompts import HYDE_PROMPT
from ..config.settings import settings

logger = logging.getLogger(__name__)


def is_ambiguous_query(question: str) -> bool:
    """
    Determina si una pregunta es ambigua y podría beneficiarse de HYDE.
    
    Args:
        question: Pregunta del usuario
        
    Returns:
        True si la pregunta contiene palabras clave ambiguas
    """
    ambiguous_keywords = ["idiomas", "lenguas", "simbolos", "que dice", "como es", "donde esta"]
    return any(keyword in question.lower() for keyword in ambiguous_keywords)


def get_docs_with_hyde(
    question: str, 
    base_retriever: BaseRetriever, 
    llm: BaseLanguageModel
) -> List[Document]:
    """
    Obtiene documentos usando la técnica HYDE (Hypothetical Document Embeddings).
    
    Args:
        question: Pregunta del usuario
        base_retriever: Retriever base para búsqueda
        llm: Modelo de lenguaje para generar documento hipotético
        
    Returns:
        Lista de documentos recuperados usando HYDE, o lista vacía si hay error
    """
    try:
        hyde_prompt = HYDE_PROMPT.format(question=question)
        hypothetical_doc_message = llm.invoke(hyde_prompt)
        hypothetical_doc = hypothetical_doc_message.content if hasattr(hypothetical_doc_message, 'content') else str(hypothetical_doc_message)
        
        hyde_docs = base_retriever.invoke(hypothetical_doc)
        return hyde_docs
    except Exception as e:
        logger.warning(f"Error en HYDE retrieval: {e}")
        return []


def retrieve_documents(
    question: str, 
    base_retriever: BaseRetriever, 
    llm: Optional[BaseLanguageModel] = None,
    use_hyde: bool = True,
    top_k: Optional[int] = None
) -> RetrievalResult:
    """
    Recupera documentos relevantes para una pregunta usando búsqueda directa o HYDE.
    
    Args:
        question: Pregunta del usuario
        base_retriever: Retriever base para búsqueda vectorial
        llm: Modelo de lenguaje (opcional, necesario para HYDE)
        use_hyde: Si usar técnica HYDE cuando la pregunta es ambigua
        top_k: Número máximo de documentos a recuperar
        
    Returns:
        RetrievalResult con documentos recuperados y método usado
    """
    final_top_k = top_k or settings.DEFAULT_TOP_K
    
    docs = base_retriever.invoke(question)
    
    retrieval_method = "direct"
    
    if use_hyde and llm and (len(docs) < settings.MIN_DOCS_FOR_HYDE or is_ambiguous_query(question)):
        hyde_docs = get_docs_with_hyde(question, base_retriever, llm)
        
        if hyde_docs:
            all_docs = docs + hyde_docs
            seen = set()
            unique_docs = []
            for doc in all_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
            docs = unique_docs[:final_top_k]
            retrieval_method = "hyde_combined"
    
    return RetrievalResult(
        documents=docs,
        query=question,
        retrieval_method=retrieval_method
    )


def create_retrieval_chain(
    base_retriever: BaseRetriever, 
    llm: Optional[BaseLanguageModel] = None, 
    use_hyde: bool = True, 
    top_k: Optional[int] = None
) -> Runnable:
    """
    Crea una cadena de recuperación de documentos para usar en pipelines.
    
    Args:
        base_retriever: Retriever base para búsqueda vectorial
        llm: Modelo de lenguaje (opcional, necesario para HYDE)
        use_hyde: Si usar técnica HYDE cuando sea apropiado
        top_k: Número máximo de documentos a recuperar
        
    Returns:
        RunnableLambda que recupera documentos para una pregunta
    """
    def retrieval_step(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        question = input_dict["question"]
        result = retrieve_documents(question, base_retriever, llm, use_hyde, top_k)
        return {
            **input_dict,
            "retrieved_docs": result.documents,
            "retrieval_method": result.retrieval_method
        }
    
    return RunnableLambda(retrieval_step)
