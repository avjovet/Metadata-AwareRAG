from typing import Dict, Any, List
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableLambda, 
    RunnableBranch, 
    RunnableParallel,
    Runnable
)
from operator import itemgetter

from langchain_core.documents import Document
from ..io.vectordb import get_vector_store, get_self_query_retriever
from ..io.llm import get_llm
from ..steps.retrieval import create_retrieval_chain
from ..steps.rerank import create_reranker
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
from ..types import PipelineInput, PipelineOutput, SemanticRouterOutput, ExtractedFilters
from ..config.settings import settings


def create_dynamic_rag_pipeline(
    db_folder_name: str,
    embedding_model_name: str,
    llm_model_name: str = None,
    temperature: float = None,
    top_k: int = None,
    enable_self_query: bool = None
) -> Runnable:
    llm_model_name = llm_model_name or "llama3.1:8b"
    temperature = temperature or 0.0
    top_k = top_k or 15
    enable_self_query = enable_self_query if enable_self_query is not None else True
    
    vector_store = get_vector_store(db_folder_name, embedding_model_name)
    llm = get_llm(model_name=llm_model_name, temperature=temperature)
    reranker = create_reranker()
    
    modular_components = create_modular_self_query_pipeline(llm, vector_store, top_k=top_k)
    
    self_query_retriever = create_self_query_retriever(llm, vector_store, top_k=top_k)
    
    def debug_quality_router(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("question", "NO_QUESTION")
        
        try:
            quality_router = create_quality_router(llm)
            quality_result = quality_router.invoke({"question": question})
            
            if isinstance(quality_result, dict):
                merged_result = {**inputs, **quality_result}
                return merged_result
            else:
                return inputs
                
        except Exception as e:
            return {
                **inputs,
                "has_spelling_errors": False,
                "corrected_question": None,
                "correction_notes": f"Error en corrección: {e}"
            }

    quality_router_chain = RunnableLambda(debug_quality_router)

    def rerank_docs(inputs: Dict[str, Any]) -> List[Document]:
        question = inputs.get("question", "NO_QUESTION")
        
        if "retrieved_docs" not in inputs:
            return []
        
        docs = inputs["retrieved_docs"]
        
        if not docs:
            return []
        
        try:
            from ..steps.rerank import rerank_documents
            rerank_result = rerank_documents(question, docs, reranker)
            return rerank_result.documents
        except Exception as e:
            return docs

    rerank_chain = RunnableLambda(rerank_docs)

    rag_answer_chain = create_rag_answer_chain(llm)
    
    def modular_self_query_with_debug(inputs: Dict[str, Any]) -> List[Document]:
        question = inputs.get("question", "NO_QUESTION")
        
        try:
            try:
                semantic_result = modular_components["semantic_router"].invoke(inputs)
            except Exception as e:
                semantic_result = {
                    **inputs,
                    "semantic_category": "general",
                    "semantic_confidence": 0.5,
                    "semantic_reasoning": f"Error en clasificación: {e}"
                }
            
            try:
                filter_result = modular_components["filter_extractor"].invoke(semantic_result)
            except Exception as e:
                filter_result = {
                    **semantic_result,
                    "extracted_filters": ExtractedFilters()
                }
            
            try:
                docs = modular_components["retrieval_assembler"].invoke(filter_result)
                return docs
            except Exception as e:
                raise e
                
        except Exception as e:
            try:
                docs = self_query_retriever.invoke(question)
                return docs
            except Exception as e2:
                try:
                    basic_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
                    docs = basic_retriever.invoke(question)
                    return docs
                except Exception as e3:
                    return []
    
    self_query_chain = RunnableLambda(modular_self_query_with_debug)
    
    final_chain = (
        RunnablePassthrough.assign(original_question=itemgetter("question"))
        | quality_router_chain
        | RunnablePassthrough.assign(retrieved_docs=self_query_chain)
        | RunnablePassthrough.assign(retrieved_docs=rerank_chain)
        | RunnablePassthrough.assign(
            generated_answer=RunnableLambda(lambda x: {
                "context": "\n\n".join([d.page_content for d in x.get("retrieved_docs", [])]) if x.get("retrieved_docs", []) else "",
                "question": x.get("question", "")
            })
            | RunnableLambda(lambda x: {
                "context": x["context"],
                "question": x["question"],
                "debug_context_length": len(x["context"]),
                "debug_context_preview": x["context"][:500] + "..." if len(x["context"]) > 500 else x["context"]
            })
            | RunnableLambda(lambda x: {
                "context": x["context"],
                "question": x["question"]
            })
            | RunnableLambda(lambda x: {
                "context": x["context"],
                "question": x["question"],
                "debug_step4_context": x["context"],
                "debug_step4_question": x["question"]
            })
            | rag_answer_chain
        )
    )

    def format_output(chain_result: Dict) -> PipelineOutput:
        retrieved_docs = chain_result.get("retrieved_docs", [])
        
        route_quality_value = None
        if "has_spelling_errors" in chain_result:
            has_errors = chain_result.get("has_spelling_errors", False)
            route_quality_value = "mal_redactada" if has_errors else "simple"

        output = PipelineOutput(
            question=chain_result["original_question"],
            generated_answer=chain_result["generated_answer"],
            retrieved_context=[doc.page_content for doc in retrieved_docs],
            route_quality=route_quality_value,
            route="simplified",
            corrected_question=chain_result.get("question") if chain_result.get("has_spelling_errors") else None
        )
        
        return output
        
    return final_chain | RunnableLambda(format_output)


def invoke_dynamic_pipeline(chain: Runnable, question: str) -> PipelineOutput:
    if not question:
        return PipelineOutput(
            question=question,
            generated_answer="",
            retrieved_context=[],
            error="La pregunta no puede estar vacía."
        )
    
    try:
        result = chain.invoke({"question": question})
        return result
    except Exception as e:
        return PipelineOutput(
            question=question,
            generated_answer=f"Error en el pipeline: {str(e)}",
            retrieved_context=[],
            error=str(e)
        )
