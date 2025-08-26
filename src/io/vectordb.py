import torch
from pathlib import Path
from typing import Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_community.query_constructors.chroma import ChromaTranslator

from ..config.settings import settings


def get_embedding_function(model_name: str) -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )


def get_vector_store(db_folder_name: str, embedding_model_name: str) -> Chroma:
    persist_path = Path(settings.CHROMA_PERSIST_PATH) / db_folder_name
    
    if not persist_path.is_dir() or not (persist_path / "chroma.sqlite3").exists():
        raise FileNotFoundError(
            f"La base de datos especificada no se encontró en '{persist_path}'. "
            "Asegúrate de que el nombre de la carpeta es correcto y la base de datos ha sido indexada."
        )
    
    embedding_function = get_embedding_function(embedding_model_name)
    return Chroma(persist_directory=str(persist_path), embedding_function=embedding_function)


def get_self_query_retriever(
    vector_store: Chroma, 
    llm, 
    enable_self_query: bool = True
) -> Optional[SelfQueryRetriever]:
    if not enable_self_query:
        return None
    
    try:
        metadata_field_info = [
            AttributeInfo(
                name="number",
                description="""El número del artículo constitucional (como string). 
                Ejemplos: '1', '2', '43', '139', '200', '206'.
                IMPORTANTE: Solo usar filtros simples con eq('number', 'X') para artículos específicos.
                Para consultas sobre múltiples artículos, usar solo búsqueda semántica sin filtros.
                Evitar filtros complejos con 'and', 'or' o comparaciones de rango.""",
                type="string",
            ),
        ]
        
        document_content_description = "Artículo de la Constitución Política del Perú con su contenido legal"
        
        return SelfQueryRetriever.from_llm(
            llm,
            vector_store,
            document_content_description,
            metadata_field_info,
            structured_query_translator=ChromaTranslator(),
            enable_limit=True,
            search_kwargs={"k": settings.DEFAULT_TOP_K}
        )
    except Exception as e:
        return None
