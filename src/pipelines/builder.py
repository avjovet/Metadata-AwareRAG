from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from ..types import PipelineInput, PipelineOutput
from .naive import create_naive_rag_pipeline, invoke_naive_pipeline
from .dinamic import create_dynamic_rag_pipeline, invoke_dynamic_pipeline


class BasePipeline(ABC):
    @abstractmethod
    def invoke(self, question: str) -> PipelineOutput:
        pass


class NaiveRAGPipeline(BasePipeline):
    def __init__(
        self,
        db_folder_name: str,
        embedding_model_name: str,
        llm_model_name: str = None,
        temperature: float = None,
        top_k: int = None
    ):
        self.chain = create_naive_rag_pipeline(
            db_folder_name=db_folder_name,
            embedding_model_name=embedding_model_name,
            llm_model_name=llm_model_name,
            temperature=temperature,
            top_k=top_k
        )
    
    def invoke(self, question: str) -> PipelineOutput:
        return invoke_naive_pipeline(self.chain, question)


class DynamicRoutedRAGPipeline(BasePipeline):
    def __init__(
        self,
        db_folder_name: str,
        embedding_model_name: str,
        llm_model_name: str = None,
        temperature: float = None,
        top_k: int = None,
        enable_self_query: bool = None
    ):
        self.chain = create_dynamic_rag_pipeline(
            db_folder_name=db_folder_name,
            embedding_model_name=embedding_model_name,
            llm_model_name=llm_model_name,
            temperature=temperature,
            top_k=top_k,
            enable_self_query=enable_self_query
        )
    
    def invoke(self, question: str) -> PipelineOutput:
        return invoke_dynamic_pipeline(self.chain, question)


def create_pipeline(
    pipeline_type: str,
    db_folder_name: str,
    embedding_model_name: str,
    **kwargs
) -> BasePipeline:
    if pipeline_type.lower() == "naive":
        return NaiveRAGPipeline(
            db_folder_name=db_folder_name,
            embedding_model_name=embedding_model_name,
            **kwargs
        )
    elif pipeline_type.lower() == "dynamic":
        return DynamicRoutedRAGPipeline(
            db_folder_name=db_folder_name,
            embedding_model_name=embedding_model_name,
            **kwargs
        )
    else:
        raise ValueError(f"Tipo de pipeline no soportado: {pipeline_type}. Use 'naive' o 'dynamic'")
