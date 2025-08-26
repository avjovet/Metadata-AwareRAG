from langchain_ollama import ChatOllama
from ..config.settings import settings


def get_llm(
    model_name: str = None, 
    temperature: float = None, 
    base_url: str = None
) -> ChatOllama:
    return ChatOllama(
        model=model_name or settings.OLLAMA_MODEL,
        base_url=base_url or settings.OLLAMA_URL,
        temperature=temperature or settings.DEFAULT_TEMPERATURE
    )


def get_llm_with_structured_output(llm: ChatOllama, output_class):
    return llm.with_structured_output(output_class)
