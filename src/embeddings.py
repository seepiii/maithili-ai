import openai
from src.config import settings

_client: openai.OpenAI | None = None

def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=settings.openai_api_key)
    return _client

def _embed(text: str) -> list[float]:
    response = _get_client().embeddings.create(model=settings.embedding_model, input=text)
    return response.data[0].embedding

def get_query_embedding(text: str) -> list[float]:
    return _embed(text)

def get_passage_embedding(text: str) -> list[float]:
    return _embed(text)
