# chat.py

from src.chatbot_logic import Chatbot
from src.pipelines import DynamicRoutedRAGPipeline

def main():
    """
    Función principal para iniciar el chatbot interactivo desde la línea de comandos.
    """
    try:
        rag_pipeline = DynamicRoutedRAGPipeline(
            db_folder_name="db_BAAI_bge-m3_json_metadata",
            embedding_model_name="BAAI/bge-m3", 
            llm_model_name="llama3.1:8b", 
            temperature=0.0,  
            top_k=15,  
            enable_self_query=True  
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