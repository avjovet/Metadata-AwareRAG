
from src.pipelines import BasePipeline, DynamicRoutedRAGPipeline

class Chatbot:
    """
    Gestiona la lógica de la conversación y la interacción con el pipeline de RAG.
    """
    def __init__(self, pipeline: BasePipeline = None):
        """
        Inicializa el chatbot con una estrategia de pipeline específica.
        Si no se proporciona ninguna, usa DynamicRoutedRAGPipeline por defecto.
        """
        print("--- Inicializando el Chatbot ---")
        if pipeline is None:
            print("INFO: No se proporcionó un pipeline. Usando 'DynamicRoutedRAGPipeline' por defecto.")
            self.pipeline = DynamicRoutedRAGPipeline(
                db_folder_name="db_BAAI_bge-m3_cs1024_co100",
                embedding_model_name="BAAI/bge-m3",
                llm_model_name="llama3.1:8b",
                temperature=0.0,
                top_k=15,
                enable_self_query=True
            )
        else:
            self.pipeline = pipeline
        print("--- Chatbot listo para conversar ---")

    def get_response(self, user_input: str) -> str:
        """
        Procesa la entrada del usuario y devuelve una respuesta formateada.
        """
        if not user_input:
            return "Por favor, escribe una pregunta."

        if user_input.lower() in ['hola', 'buenos días', 'buenas tardes']:
            return "¡Hola! ¿En qué puedo ayudarte hoy?"
        
        if user_input.lower() in ['gracias', 'muchas gracias']:
            return "De nada. ¡Estoy aquí para ayudar!"

        print(f"DEBUG: Enviando la pregunta al pipeline: '{user_input}'")
        rag_output = self.pipeline.invoke(user_input)

        if rag_output.error:
            return f"Lo siento, ocurrió un error: {rag_output.error}"

        answer = rag_output.generated_answer or "No pude encontrar una respuesta."
        
        return answer
