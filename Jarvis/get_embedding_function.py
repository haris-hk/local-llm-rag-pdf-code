from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    # Local embedding via Ollama; change model if you prefer a different embedding model.
    return OllamaEmbeddings(model="nomic-embed-text")
