"""Validadores centralizados para inputs del sistema."""


class QuestionValidator:
    """Validador centralizado para preguntas de usuarios."""
    
    MIN_LENGTH = 3
    MAX_LENGTH = 500
    
    @staticmethod
    def validate(question: str | None) -> tuple[bool, str | None]:
        """
        Valida una pregunta del usuario.
        
        Args:
            question: Pregunta a validar (puede ser None)
            
        Returns:
            Tuple de (is_valid, error_message)
            - is_valid: True si la pregunta es válida, False en caso contrario
            - error_message: Mensaje de error si no es válida, None si es válida
        """
        if not question:
            return False, "La pregunta no puede estar vacía."
        
        question = question.strip()
        
        if len(question) < QuestionValidator.MIN_LENGTH:
            return False, f"La pregunta debe tener al menos {QuestionValidator.MIN_LENGTH} caracteres."
        
        if len(question) > QuestionValidator.MAX_LENGTH:
            return False, f"La pregunta no puede exceder {QuestionValidator.MAX_LENGTH} caracteres."
        
        return True, None

