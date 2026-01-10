import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# 1. Configuración de parámetros
PDF_PATH = "mesicic2_per_const_sp.pdf"  # Tu archivo de la Constitución
OUTPUT_FILE = "chunks_semanticos_nomic.json"
MODEL_NAME = "nomic-embed-text:latest"

def generate_semantic_json():
    # 2. Cargar el modelo de Nomic (asegúrate de que Ollama esté corriendo)
    print(f"Cargando modelo {MODEL_NAME}...")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    # 3. Cargar y extraer texto del PDF
    print("Extrayendo texto del PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    # Unimos todo el texto para que el chunker semántico analice el flujo completo
    full_text = " ".join([p.page_content for p in pages])

    # 4. Configurar el Chunker Semántico
    # 'percentile' identifica los puntos donde el tema cambia drásticamente
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )

    # 5. Ejecutar la fragmentación
    print("Realizando fragmentación semántica... (esto puede tardar un poco)")
    docs = chunker.create_documents([full_text])

    # 6. Estructurar la data para el JSON
    data_to_export = []
    for i, doc in enumerate(docs):
        chunk_entry = {
            "chunk_id": i + 1,
            "content": doc.page_content,
            "metadata": {
                "source": PDF_PATH,
                "strategy": "Semantic Chunking",
                "model": MODEL_NAME,
                "char_length": len(doc.page_content)
            }
        }
        data_to_export.append(chunk_entry)

    # 7. Guardar en archivo JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_to_export, f, ensure_ascii=False, indent=4)

    print(f"¡Éxito! Se han guardado {len(data_to_export)} chunks en {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_semantic_json()