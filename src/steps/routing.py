from typing import Dict, Any
from langchain_core.runnables import RunnableLambda

from ..types import QualityRouterOutput, MainRouterOutput, SubQuestionsOutput, StepBackOutput
from ..steps.prompts import (
    QUALITY_ROUTER_SYSTEM_PROMPT,
    MAIN_ROUTER_SYSTEM_PROMPT,
    DECOMPOSITION_SYSTEM_PROMPT,
    STEP_BACK_SYSTEM_PROMPT
)


def create_quality_router(llm):
    def quality_router_step(x: Dict[str, Any]) -> Dict[str, Any]:
        try:
            raw_response = llm.invoke([
                ("system", QUALITY_ROUTER_SYSTEM_PROMPT),
                ("human", f"Analiza esta pregunta: '{x['question']}'")
            ])
            
            import json
            try:
                result_dict = json.loads(raw_response.content.strip())
                has_errors = result_dict.get("has_spelling_errors", False)
                corrected_question = result_dict.get("corrected_question", None)
                
                final_question = corrected_question if (has_errors and corrected_question) else x['question']
                
                return {
                    **x, 
                    "question": final_question,
                    "original_question": x['question'],
                    "has_spelling_errors": has_errors
                }
                
            except json.JSONDecodeError as json_error:
                raise json_error
            
        except Exception:
            return {
                **x, 
                "question": x['question'],
                "original_question": x['question'],
                "has_spelling_errors": False
            }
    
    return RunnableLambda(quality_router_step)


def create_main_router(llm):
    main_router_llm = llm.with_structured_output(MainRouterOutput)
    
    def main_router_step(x: Dict[str, Any]) -> Dict[str, Any]:
        result = main_router_llm.invoke([
            ("system", MAIN_ROUTER_SYSTEM_PROMPT),
            ("human", f"Clasifica esta pregunta: '{x['question']}'")
        ])
        return {**x, "route": result.route}
    
    return RunnableLambda(main_router_step)


def create_decomposition_chain(llm):
    decomp_llm = llm.with_structured_output(SubQuestionsOutput)
    
    def decomposition_step(x: Dict[str, Any]) -> Dict[str, Any]:
        result = decomp_llm.invoke([
            ("system", DECOMPOSITION_SYSTEM_PROMPT),
            ("human", f"Descompón esta pregunta compleja: '{x['question']}'")
        ])
        return {**x, "sub_questions": result.sub_questions}
    
    return RunnableLambda(decomposition_step)


def create_step_back_generator(llm):
    stepback_llm = llm.with_structured_output(StepBackOutput)
    
    def step_back_generation_step(x: Dict[str, Any]) -> Dict[str, Any]:
        result = stepback_llm.invoke([
            ("system", STEP_BACK_SYSTEM_PROMPT),
            ("human", f"Genera una pregunta step-back para: '{x['question']}'")
        ])
        return {**x, "step_back_question": result.step_back_question}
    
    return RunnableLambda(step_back_generation_step)