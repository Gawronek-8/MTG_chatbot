from langchain_ollama import OllamaEmbeddings
from config import settings



ollama_embedder = OllamaEmbeddings(
    model=settings.ollama_embed_model,
    base_url=settings.ollama_url
)


def calculate_expected_tokens(text: str) -> int:
    """
    Calculate the expected number of tokens for a given text using the Ollama embedder.

    Args:
        text (str): The input text to calculate tokens for.
    """

    return (len(text) // 4) + 1