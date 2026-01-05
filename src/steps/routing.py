import json
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda

from ..types import QualityRouterOutput, MainRouterOutput, SubQuestionsOutput, StepBackOutput
from ..steps.prompts import (
    QUALITY_ROUTER_SYSTEM_PROMPT,
    MAIN_ROUTER_SYSTEM_PROMPT,
    DECOMPOSITION_SYSTEM_PROMPT,
    STEP_BACK_SYSTEM_PROMPT
)


def create_quality_router(llm: BaseLanguageModel) -> Runnable:
    """
    Crea un router de calidad para detectar y corregir errores ortográficos en preguntas.
    
    Args:
        llm: Modelo de lenguaje para análisis de calidad
        
    Returns:
        Runnable que procesa preguntas y corrige errores ortográficos si los hay
    """
    def quality_router_step(x: Dict[str, Any]) -> Dict[str, Any]:
        try:
            raw_response = llm.invoke([
                ("system", QUALITY_ROUTER_SYSTEM_PROMPT),
                ("human", f"Analiza esta pregunta: '{x['question']}'")
            ])
            
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
                logger.warning(f"Error parsing JSON en quality router: {json_error}")
                raise json_error
            
        except Exception as e:
            logger.error(f"Error en quality router: {e}", exc_info=True)
            return {
                **x, 
                "question": x['question'],
                "original_question": x['question'],
                "has_spelling_errors": False
            }
    
    return RunnableLambda(quality_router_step)


def create_main_router(llm: BaseLanguageModel) -> Runnable:
    """
    Crea un router principal para clasificar el tipo de pregunta.
    
    Args:
        llm: Modelo de lenguaje para clasificación
        
    Returns:
        Runnable que clasifica preguntas como 'simple', 'compleja' o 'step_back'
    """
    main_router_llm = llm.with_structured_output(MainRouterOutput)
    
    def main_router_step(x: Dict[str, Any]) -> Dict[str, Any]:
        result = main_router_llm.invoke([
            ("system", MAIN_ROUTER_SYSTEM_PROMPT),
            ("human", f"Clasifica esta pregunta: '{x['question']}'")
        ])
        return {**x, "route": result.route}
    
    return RunnableLambda(main_router_step)


def create_decomposition_chain(llm: BaseLanguageModel) -> Runnable:
    """
    Crea una cadena para descomponer preguntas complejas en sub-preguntas.
    
    Args:
        llm: Modelo de lenguaje para descomposición
        
    Returns:
        Runnable que genera sub-preguntas a partir de una pregunta compleja
    """
    decomp_llm = llm.with_structured_output(SubQuestionsOutput)
    
    def decomposition_step(x: Dict[str, Any]) -> Dict[str, Any]:
        result = decomp_llm.invoke([
            ("system", DECOMPOSITION_SYSTEM_PROMPT),
            ("human", f"Descompón esta pregunta compleja: '{x['question']}'")
        ])
        return {**x, "sub_questions": result.sub_questions}
    
    return RunnableLambda(decomposition_step)


def create_step_back_generator(llm: BaseLanguageModel) -> Runnable:
    """
    Crea un generador de preguntas step-back para razonamiento de alto nivel.
    
    Args:
        llm: Modelo de lenguaje para generación de preguntas step-back
        
    Returns:
        Runnable que genera preguntas más generales basadas en principios fundamentales
    """
    stepback_llm = llm.with_structured_output(StepBackOutput)
    
    def step_back_generation_step(x: Dict[str, Any]) -> Dict[str, Any]:
        result = stepback_llm.invoke([
            ("system", STEP_BACK_SYSTEM_PROMPT),
            ("human", f"Genera una pregunta step-back para: '{x['question']}'")
        ])
        return {**x, "step_back_question": result.step_back_question}
    
    return RunnableLambda(step_back_generation_step)