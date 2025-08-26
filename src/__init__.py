# src/__init__.py
# This file makes the src directory a Python package

# Exportar las interfaces principales
from .pipelines.builder import create_pipeline, BasePipeline, NaiveRAGPipeline, DynamicRoutedRAGPipeline
from .types import PipelineInput, PipelineOutput

__all__ = [
    'create_pipeline',
    'BasePipeline', 
    'NaiveRAGPipeline',
    'DynamicRoutedRAGPipeline',
    'PipelineInput',
    'PipelineOutput'
]
