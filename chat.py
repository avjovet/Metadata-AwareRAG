# chat.py

import os
from src.chatbot_logic import Chatbot
from src.pipelines import DynamicRoutedRAGPipeline
from src.config.settings import settings, get_default_db_folder_name

def main():
    """
    Función principal para iniciar el chatbot interactivo desde la línea de comandos.
    """
    try:
        # Construir db_folder_name desde settings (formato JSON)
        default_db_name = get_default_db_folder_name(
            embedding_model=settings.EMBEDDER_MODEL,
            db_identifier=settings.DEFAULT_DB_IDENTIFIER
        )
        
        rag_pipeline = DynamicRoutedRAGPipeline(
            db_folder_name=os.getenv("DEFAULT_DB_FOLDER_NAME", default_db_name),
            embedding_model_name=settings.EMBEDDER_MODEL,
            llm_model_name=settings.OLLAMA_MODEL,
            temperature=settings.DEFAULT_TEMPERATURE,
            top_k=settings.DYNAMIC_PIPELINE_TOP_K,
            enable_self_query=settings.ENABLE_SELF_QUERY
        )
        
        chatbot = Chatbot(pipeline=rag_pipeline)
        
        while True:
            user_input = input("Tú: ")
            
            if user_input.lower() == 'salir':
                break
                
            response = chatbot.get_response(user_input)
            print(f"Asistente: {response}")

    except FileNotFoundError as e:
        print(f"\nERROR CRÍTICO: {e}")
        print("Asegúrate de haber creado la base de datos vectorial antes de iniciar el chat.")
        print("Puedes hacerlo ejecutando: python -m src.indexing_logic")
        
    except Exception as e:
        print(f"\nHa ocurrido un error inesperado: {e}")

if __name__ == "__main__":
    main()