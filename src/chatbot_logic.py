import logging
import os

from src.pipelines import BasePipeline, DynamicRoutedRAGPipeline
from src.utils.validators import QuestionValidator
from src.config.settings import settings, get_default_db_folder_name

logger = logging.getLogger(__name__)


class Chatbot:
    """
    Gestiona la lógica de la conversación y la interacción con el pipeline de RAG.
    """
    def __init__(self, pipeline: BasePipeline = None):
        """
        Inicializa el chatbot con una estrategia de pipeline específica.
        Si no se proporciona ninguna, usa DynamicRoutedRAGPipeline por defecto.
        """
        logger.info("Inicializando el Chatbot")
        if pipeline is None:
            logger.info("No se proporcionó un pipeline. Usando 'DynamicRoutedRAGPipeline' por defecto.")
            
            # Construir db_folder_name desde settings (formato raw con chunking)
            default_db_name = get_default_db_folder_name(
                embedding_model=settings.EMBEDDER_MODEL,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            
            self.pipeline = DynamicRoutedRAGPipeline(
                db_folder_name=os.getenv("DEFAULT_DB_FOLDER_NAME", default_db_name),
                embedding_model_name=settings.EMBEDDER_MODEL,
                llm_model_name=settings.OLLAMA_MODEL,
                temperature=settings.DEFAULT_TEMPERATURE,
                top_k=settings.DYNAMIC_PIPELINE_TOP_K,
                enable_self_query=settings.ENABLE_SELF_QUERY
            )
        else:
            self.pipeline = pipeline
        logger.info("Chatbot listo para conversar")

    def get_response(self, user_input: str) -> str:
        """
        Procesa la entrada del usuario y devuelve una respuesta formateada.
        
        Args:
            user_input: Entrada del usuario
            
        Returns:
            Respuesta formateada del chatbot
        """
        is_valid, error_message = QuestionValidator.validate(user_input)
        if not is_valid:
            return error_message or "Por favor, escribe una pregunta válida."

        if user_input.lower() in ['hola', 'buenos días', 'buenas tardes']:
            return "¡Hola! ¿En qué puedo ayudarte hoy?"
        
        if user_input.lower() in ['gracias', 'muchas gracias']:
            return "De nada. ¡Estoy aquí para ayudar!"

        logger.debug(f"Enviando la pregunta al pipeline: '{user_input}'")
        rag_output = self.pipeline.invoke(user_input)

        if rag_output.error:
            return f"Lo siento, ocurrió un error: {rag_output.error}"

        answer = rag_output.generated_answer or "No pude encontrar una respuesta."
        
        return answer
