from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.retrievers import BaseRetriever

from ..types import RetrievalResult
from ..steps.prompts import HYDE_PROMPT
from ..config.settings import settings


def docs_to_text(docs: List[Document]) -> str:
    return "\n\n".join([d.page_content for d in docs])


def is_ambiguous_query(question: str) -> bool:
    ambiguous_keywords = ["idiomas", "lenguas", "simbolos", "que dice", "como es", "donde esta"]
    return any(keyword in question.lower() for keyword in ambiguous_keywords)


def get_docs_with_hyde(question: str, base_retriever: BaseRetriever, llm) -> List[Document]:
    try:
        hyde_prompt = HYDE_PROMPT.format(question=question)
        hypothetical_doc_message = llm.invoke(hyde_prompt)
        hypothetical_doc = hypothetical_doc_message.content if hasattr(hypothetical_doc_message, 'content') else str(hypothetical_doc_message)
        
        hyde_docs = base_retriever.invoke(hypothetical_doc)
        return hyde_docs
    except Exception:
        return []


def retrieve_documents(
    question: str, 
    base_retriever: BaseRetriever, 
    llm=None,
    use_hyde: bool = True,
    top_k: int = None
) -> RetrievalResult:
    final_top_k = top_k or settings.DEFAULT_TOP_K
    
    docs = base_retriever.invoke(question)
    
    retrieval_method = "direct"
    
    if use_hyde and llm and (len(docs) < 5 or is_ambiguous_query(question)):
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


def create_retrieval_chain(base_retriever: BaseRetriever, llm=None, use_hyde: bool = True, top_k: int = None):
    def retrieval_step(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        question = input_dict["question"]
        result = retrieve_documents(question, base_retriever, llm, use_hyde, top_k)
        return {
            **input_dict,
            "retrieved_docs": result.documents,
            "retrieval_method": result.retrieval_method
        }
    
    return RunnableLambda(retrieval_step)
