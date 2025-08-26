from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from ..steps.prompts import (
    RAG_OPTIMIZED_PROMPT,
    RAG_OPTIMIZED_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    COMPLEX_PROMPT,
    STEP_BACK_PROMPT
)


def docs_to_text(docs: List[Document]) -> str:
    return "\n\n".join([d.page_content for d in docs])


def create_rag_answer_chain(llm):
    return (
        {
            "context": lambda x: x.get("context", ""),
            "question": lambda x: x.get("question", "")
        } | RunnableLambda(lambda x: [
            ("system", RAG_OPTIMIZED_SYSTEM_PROMPT),
            ("human", RAG_OPTIMIZED_PROMPT.format(**x))
        ]) | llm | StrOutputParser()
    )


def create_synthesis_chain(llm):
    return SYNTHESIS_PROMPT | llm | StrOutputParser()


def create_complex_answer_chain(llm):
    return COMPLEX_PROMPT | llm | StrOutputParser()


def create_step_back_answer_chain(llm):
    return STEP_BACK_PROMPT | llm | StrOutputParser()


def process_complex_question(x: Dict[str, Any], llm, retrieval_func) -> Dict[str, Any]:
    sub_questions = x["sub_questions"]
    original_question = x["original_question"]
    
    expanded_query = f"{original_question} {' '.join(sub_questions)}"
    
    retrieved_docs = retrieval_func(expanded_query)
    
    topics_checklist = "\n".join([f"- {sq}" for sq in sub_questions])
    
    complex_prompt = f"""
Responde la pregunta principal de forma concisa y directa, basándote únicamente en el contexto proporcionado.

REGLAS ESTRICTAS:
- Sintetiza la información para cubrir los siguientes aspectos clave, sin añadir detalles extra:
{topics_checklist}
- NO incluyas introducciones, conclusiones ni información no solicitada.
- Estructura la respuesta de forma clara, pero breve.
- Cita artículos específicos cuando sea posible.
- Si la respuesta a alguno de los aspectos no está en el contexto, omítelo.

Contexto:
{docs_to_text(retrieved_docs)}

Pregunta: {original_question}

Respuesta Concisa y Estructurada:"""
    
    generated_answer = (COMPLEX_PROMPT | llm | StrOutputParser()).invoke({
        "context": docs_to_text(retrieved_docs),
        "question": original_question,
        "topics_checklist": topics_checklist
    })
    
    return {
        "generated_answer": generated_answer,
        "retrieved_docs": retrieved_docs,
        "sub_questions": sub_questions
    }


def create_complex_branch_chain(llm, retrieval_func):
    def complex_step(x: Dict[str, Any]) -> Dict[str, Any]:
        result = process_complex_question(x, llm, retrieval_func)
        return {
            **x,
            "generated_answer": result["generated_answer"],
            "retrieved_docs": result["retrieved_docs"]
        }
    
    return RunnableLambda(complex_step)


def create_step_back_branch_chain(llm, retrieval_func):
    def step_back_step(x: Dict[str, Any]) -> Dict[str, Any]:
        step_back_question = x["step_back_question"]
        original_question = x["original_question"]
        
        contexts = RunnableParallel(
            normal_context=lambda x: retrieval_func(x["question"]),
            step_back_context=lambda x: retrieval_func(x["step_back_question"]),
        ).invoke({
            "question": original_question,
            "step_back_question": step_back_question
        })
        
        generated_answer = create_step_back_answer_chain(llm).invoke({
            "question": original_question, 
            "normal_context": docs_to_text(contexts["normal_context"]),
            "step_back_context": docs_to_text(contexts["step_back_context"])
        })
        
        all_docs = contexts["normal_context"] + contexts["step_back_context"]
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        
        return {
            **x,
            "generated_answer": generated_answer,
            "retrieved_docs": unique_docs
        }
    
    return RunnableLambda(step_back_step)
