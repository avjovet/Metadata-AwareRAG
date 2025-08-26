from typing import Dict, Any
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from ..io.vectordb import get_vector_store
from ..io.llm import get_llm
from ..steps.prompts import RAG_BASIC_PROMPT
from ..steps.retrieval import docs_to_text
from ..types import PipelineInput, PipelineOutput


def create_naive_rag_pipeline(
    db_folder_name: str,
    embedding_model_name: str,
    llm_model_name: str = None,
    temperature: float = None,
    top_k: int = None
):
    llm_model_name = llm_model_name or "llama3.1:8b"
    temperature = temperature or 0.1
    top_k = top_k or 5
    
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


def invoke_naive_pipeline(chain, question: str) -> PipelineOutput:
    if not question:
        return PipelineOutput(
            question=question,
            generated_answer="",
            retrieved_context=[],
            error="La pregunta no puede estar vacía."
        )
    
    output = chain.invoke({"question": question})
    
    return PipelineOutput(
        question=question,
        generated_answer=output["answer"],
        retrieved_context=[doc.page_content for doc in output["original_docs"]]
    )
