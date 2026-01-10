"""
Script para indexar los diferentes tipos de chunking en bases de datos vectoriales.
Usa la lógica existente del proyecto pero adaptada para múltiples archivos en una BD.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List

sys.path.append(str(Path(__file__).parent))

from src.indexing_logic import get_embedding_model
from src.config.settings import settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuración común
embedding_model_name = "BAAI/bge-m3"
force_reindex = True  # Cambiar a False si ya están indexadas

# Usar un chunk_size grande para evitar dividir chunks que ya están listos
chunk_size = 10000  # Tamaño grande para evitar divisiones innecesarias
chunk_overlap = 0   # Sin overlap ya que los chunks ya están definidos

def index_multiple_json_files(
    json_file_paths: List[str],
    embedding_model_name: str,
    db_identifier: str,
    force_reindex: bool = False,
    chunk_size: int = 10000,
    chunk_overlap: int = 0
):
    """
    Indexa múltiples archivos JSON en una sola base de datos vectorial.
    Si la BD ya existe y force_reindex=False, agrega los documentos a la BD existente.
    """
    safe_model_name = embedding_model_name.replace("/", "_")
    db_folder_name = f"db_{safe_model_name}_{db_identifier}"
    persist_path = Path(settings.CHROMA_PERSIST_PATH) / db_folder_name
    
    db_exists = persist_path.is_dir() and (persist_path / "chroma.sqlite3").exists()
    
    if db_exists and not force_reindex:
        print(f"  ⚠️  La BD {db_folder_name} ya existe. Usa force_reindex=True para reindexar.")
        return
    
    # Si existe y force_reindex=True, eliminar la BD anterior
    if db_exists and force_reindex:
        import shutil
        print(f"  🗑️  Eliminando BD anterior: {db_folder_name}")
        shutil.rmtree(persist_path)
    
    # Cargar y procesar todos los documentos
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    for json_file_path in json_file_paths:
        if not os.path.exists(json_file_path):
            print(f"  ⚠️  Archivo no encontrado: {json_file_path}")
            continue
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                formatted_docs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  ✗ Error al leer {json_file_path}: {e}")
            continue
        
        if not formatted_docs:
            continue
        
        # Procesar cada documento del JSON
        for i, doc in enumerate(formatted_docs):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Agregar información del archivo fuente
            metadata["source_file"] = Path(json_file_path).name
            
            if len(content) > chunk_size:
                # Dividir si es muy grande
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
                    all_documents.append(Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata
                    ))
            else:
                # Usar el chunk tal cual
                chunk_metadata = {
                    **metadata,
                    "chunk_id": f"doc_{i}_chunk_0",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "original_doc_index": i,
                    "chunk_size": len(content)
                }
                all_documents.append(Document(
                    page_content=content,
                    metadata=chunk_metadata
                ))
    
    if not all_documents:
        print(f"  ⚠️  No se encontraron documentos para indexar")
        return
    
    # Crear la base de datos vectorial
    print(f"  📊 Total de documentos a indexar: {len(all_documents)}")
    print(f"  🚀 Creando embeddings y guardando en BD...")
    
    embedding_function = get_embedding_model(embedding_model_name)
    
    Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_function,
        persist_directory=str(persist_path)
    )
    
    print(f"  ✅ BD creada: {db_folder_name}")

def index_semantic_chunks():
    """PASO 1: Indexa todos los archivos JSON semánticos de datajson_prepositional/"""
    
    print("\n" + "="*70)
    print("PASO 1: INDEXANDO CHUNKS SEMÁNTICOS")
    print("="*70)
    
    semantic_dir = Path("datajson_prepositional")
    
    if not semantic_dir.exists():
        print(f"  ✗ ERROR: El directorio {semantic_dir} no existe")
        return False
    
    json_files = sorted(semantic_dir.glob("*.json"))
    
    if not json_files:
        print(f"  ⚠️  No se encontraron archivos JSON en {semantic_dir}/")
        return False
    
    db_identifier = "semantic_chunks"
    safe_model_name = embedding_model_name.replace("/", "_")
    db_folder_name = f"db_{safe_model_name}_{db_identifier}"
    
    print(f"\n📁 Directorio: {semantic_dir}/")
    print(f"📊 Archivos encontrados: {len(json_files)}")
    print(f"💾 Base de datos: {db_folder_name}")
    
    print(f"\n📋 Archivos a indexar:")
    for i, json_file in enumerate(json_files, 1):
        file_size = json_file.stat().st_size / (1024 * 1024)  # MB
        print(f"  {i}. {json_file.name} ({file_size:.2f} MB)")
    
    print(f"\n🚀 Iniciando indexación...")
    start_time = datetime.now()
    
    json_file_paths = [str(f) for f in json_files]
    
    try:
        index_multiple_json_files(
            json_file_paths=json_file_paths,
            embedding_model_name=embedding_model_name,
            db_identifier=db_identifier,
            force_reindex=force_reindex,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        persist_path = Path(settings.CHROMA_PERSIST_PATH) / db_folder_name
        if persist_path.exists() and (persist_path / "chroma.sqlite3").exists():
            print(f"\n✅ Indexación de chunks semánticos completada")
            print(f"   ⏱️  Tiempo total: {duration:.2f} segundos")
            print(f"   📍 BD creada en: {persist_path}")
            return True
        else:
            print(f"\n⚠️  ADVERTENCIA: La BD no se creó correctamente")
            return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def index_structural_chunks():
    """PASO 2: Indexa todos los archivos JSON estructurales con metadatos de datajson/"""
    
    print("\n" + "="*70)
    print("PASO 2: INDEXANDO CHUNKS ESTRUCTURALES CON METADATOS")
    print("="*70)
    
    structural_dir = Path("datajson")
    
    if not structural_dir.exists():
        print(f"  ✗ ERROR: El directorio {structural_dir} no existe")
        return False
    
    json_files = sorted(structural_dir.glob("*.json"))
    
    if not json_files:
        print(f"  ⚠️  No se encontraron archivos JSON en {structural_dir}/")
        return False
    
    db_identifier = "structural_metadata_chunks"
    safe_model_name = embedding_model_name.replace("/", "_")
    db_folder_name = f"db_{safe_model_name}_{db_identifier}"
    
    print(f"\n📁 Directorio: {structural_dir}/")
    print(f"📊 Archivos encontrados: {len(json_files)}")
    print(f"💾 Base de datos: {db_folder_name}")
    
    print(f"\n📋 Archivos a indexar:")
    for i, json_file in enumerate(json_files, 1):
        file_size = json_file.stat().st_size / (1024 * 1024)  # MB
        print(f"  {i}. {json_file.name} ({file_size:.2f} MB)")
    
    print(f"\n🚀 Iniciando indexación...")
    start_time = datetime.now()
    
    json_file_paths = [str(f) for f in json_files]
    
    try:
        index_multiple_json_files(
            json_file_paths=json_file_paths,
            embedding_model_name=embedding_model_name,
            db_identifier=db_identifier,
            force_reindex=force_reindex,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        persist_path = Path(settings.CHROMA_PERSIST_PATH) / db_folder_name
        if persist_path.exists() and (persist_path / "chroma.sqlite3").exists():
            print(f"\n✅ Indexación de chunks estructurales completada")
            print(f"   ⏱️  Tiempo total: {duration:.2f} segundos")
            print(f"   📍 BD creada en: {persist_path}")
            return True
        else:
            print(f"\n⚠️  ADVERTENCIA: La BD no se creó correctamente")
            return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_databases():
    """PASO 3: Verifica que las bases de datos se crearon correctamente"""
    
    print("\n" + "="*70)
    print("PASO 3: VERIFICANDO BASES DE DATOS")
    print("="*70)
    
    safe_model_name = embedding_model_name.replace("/", "_")
    databases = [
        {
            "name": "Semantic Chunks",
            "db_folder": f"db_{safe_model_name}_semantic_chunks",
            "pipeline": "Naive RAG"
        },
        {
            "name": "Structural + Metadata Chunks",
            "db_folder": f"db_{safe_model_name}_structural_metadata_chunks",
            "pipeline": "Naive RAG y Dynamic Pipeline"
        }
    ]
    
    all_ok = True
    
    for db_info in databases:
        persist_path = Path(settings.CHROMA_PERSIST_PATH) / db_info["db_folder"]
        
        print(f"\n📊 {db_info['name']}")
        print(f"   BD: {db_info['db_folder']}")
        print(f"   Ruta: {persist_path}")
        
        if persist_path.exists():
            sqlite_file = persist_path / "chroma.sqlite3"
            if sqlite_file.exists():
                size_mb = sqlite_file.stat().st_size / (1024 * 1024)
                print(f"   ✅ Existe (Tamaño: {size_mb:.2f} MB)")
                print(f"   🔧 Usar con: {db_info['pipeline']}")
            else:
                print(f"   ⚠️  Directorio existe pero falta chroma.sqlite3")
                all_ok = False
        else:
            print(f"   ✗ No existe")
            all_ok = False
    
    return all_ok

def main():
    """Función principal para indexar todas las estrategias de chunking paso a paso."""
    
    print("\n" + "="*70)
    print("INDEXACIÓN DE ESTRATEGIAS DE CHUNKING PARA COMPARACIÓN")
    print("="*70)
    print(f"\n📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Modelo de embeddings: {embedding_model_name}")
    print(f"📂 Directorio de BDs: {settings.CHROMA_PERSIST_PATH}")
    print(f"⚙️  Chunk size: {chunk_size} (grande para no dividir chunks pre-procesados)")
    print(f"🔄 Force reindex: {force_reindex}")
    
    # Ejecutar pasos
    step1_ok = index_semantic_chunks()
    step2_ok = index_structural_chunks()
    step3_ok = verify_databases()
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    
    if step1_ok and step2_ok and step3_ok:
        print("\n✅ ¡TODAS LAS BASES DE DATOS SE CREARON CORRECTAMENTE!")
        print("\n📊 Bases de datos disponibles en vector_dbs/:")
        print("\n   1. db_BAAI_bge-m3_semantic_chunks")
        print("      → Usar con: NAIVE pipeline")
        print("      → Chunks semánticos (sin metadatos ricos)")
        print()
        print("   2. db_BAAI_bge-m3_structural_metadata_chunks")
        print("      → Usar con: NAIVE pipeline (ignora metadatos)")
        print("      → Usar con: DYNAMIC pipeline (usa metadatos)")
        print("      → Chunks estructurales con metadatos ricos")
    else:
        print("\n⚠️  ALGUNAS BASES DE DATOS NO SE CREARON CORRECTAMENTE")
        print("   Revisa los errores arriba")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

