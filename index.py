
import argparse
import torch
from src.indexer import create_knowledge_base
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexar documentos para RAG")
    parser.add_argument("--embedding_model", type=str, default=settings.EMBEDDER_MODEL, 
                       help="Modelo de embeddings a usar")
    parser.add_argument("--chunk_size", type=int, default=settings.CHUNK_SIZE, help="Tamaño de chunk")
    parser.add_argument("--chunk_overlap", type=int, default=settings.CHUNK_OVERLAP, help="Overlap entre chunks")
    parser.add_argument("--force_reindex", action="store_true", help="Forzar re-indexación")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 1200, "normalize_embeddings": True}
    )
    
    create_knowledge_base(
        embedding_model=embedding_model,
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_reindex=args.force_reindex
    )