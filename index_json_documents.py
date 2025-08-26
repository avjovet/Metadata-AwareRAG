"""
Script para indexar documentos JSON con metadatos en la base de datos vectorial.
"""

import sys
import os
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.indexing_logic import index_json_documents
from src.config.settings import settings


def index_all_json_documents():
    """Indexa todos los archivos JSON de la carpeta datajson con estrategia Small-to-Big."""
    
    embedding_model_name = "BAAI/bge-m3"
    db_identifier = "json_metadata"
    force_reindex = True
    
    chunk_size = 500 
    chunk_overlap = 50 
    
    json_files = [
        {
            "file": "datajson/constitucion_unificada.json",
            "description": "Constitución Política del Perú"
        },
        {
            "file": "datajson/preguntas_laborales_unificado.json", 
            "description": "Preguntas Frecuentes de Derecho Laboral"
        },
        {
            "file": "datajson/compendio_unificada.json",
            "description": "Compendio de Derecho Laboral"
        }
    ]
    
    for json_file in json_files:
        file_path = json_file["file"]
        
        if not os.path.exists(file_path):
            continue
        
        try:
            index_json_documents(
                json_file_path=file_path,
                embedding_model_name=embedding_model_name,
                db_identifier=db_identifier,
                force_reindex=force_reindex,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except Exception:
            continue

if __name__ == "__main__":
    index_all_json_documents()
