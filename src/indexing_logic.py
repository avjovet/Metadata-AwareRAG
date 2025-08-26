
from pathlib import Path
import torch
import json
from typing import List, Dict, Any, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import settings

def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Función auxiliar para inicializar el modelo de embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

def index_raw_documents(
    embedding_model_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    force_reindex: bool = False
):
    """
    Carga, divide y crea una base de datos vectorial a partir de documentos en bruto.
    Permite sobrescribir la configuración por defecto para experimentación.
    """
    final_embedding_model = embedding_model_name or settings.EMBEDDER_MODEL
    final_chunk_size = chunk_size or settings.CHUNK_SIZE
    final_chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    safe_model_name = final_embedding_model.replace("/", "_")
    db_folder_name = f"db_{safe_model_name}_cs{final_chunk_size}_co{final_chunk_overlap}"
    persist_path = Path(settings.CHROMA_PERSIST_PATH) / db_folder_name
    
    db_exists = persist_path.is_dir() and (persist_path / "chroma.sqlite3").exists()
    if db_exists and not force_reindex:
        return

    pdf_loader = DirectoryLoader(settings.DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
    txt_loader = DirectoryLoader(settings.DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    docs = pdf_loader.load() + txt_loader.load()

    if not docs:
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap
    )
    splits = text_splitter.split_documents(docs)

    embedding_function = get_embedding_model(final_embedding_model)
    Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=str(persist_path)
    )


def index_json_documents(
    json_file_path: str,
    embedding_model_name: str,
    db_identifier: str,
    force_reindex: bool = False,
    chunk_size: int = 1200,
    chunk_overlap: int = 100
):
    """
    Crea una base de datos vectorial a partir de un archivo JSON con documentos pre-formateados.
    Implementa la estrategia "Small-to-Big" con chunks pequeños que heredan metadatos del documento padre.
    """
    safe_model_name = embedding_model_name.replace("/", "_")
    db_folder_name = f"db_{safe_model_name}_{db_identifier}"
    persist_path = Path(settings.CHROMA_PERSIST_PATH) / db_folder_name

    db_exists = persist_path.is_dir() and (persist_path / "chroma.sqlite3").exists()
    if db_exists and not force_reindex:
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            formatted_docs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    if not formatted_docs:
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    all_small_chunks = []
    
    for i, doc in enumerate(formatted_docs):
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        
        if len(content) > chunk_size:
            text_chunks = text_splitter.split_text(content)
            
            for j, chunk_text in enumerate(text_chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_id": f"doc_{i}_chunk_{j}",
                    "chunk_index": j,
                    "total_chunks": len(text_chunks),
                    "original_doc_index": i,
                    "chunk_size": len(chunk_text)
                }
                
                small_doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                all_small_chunks.append(small_doc)
                
        else:
            chunk_metadata = {
                **metadata,
                "chunk_id": f"doc_{i}_chunk_0",
                "chunk_index": 0,
                "total_chunks": 1,
                "original_doc_index": i,
                "chunk_size": len(content)
            }
            
            small_doc = Document(
                page_content=content,
                metadata=chunk_metadata
            )
            all_small_chunks.append(small_doc)
    
    embedding_function = get_embedding_model(embedding_model_name)
    
    Chroma.from_documents(
        documents=all_small_chunks,
        embedding=embedding_function,
        persist_directory=str(persist_path)
    )


if __name__ == '__main__':
    index_raw_documents(force_reindex=True)
