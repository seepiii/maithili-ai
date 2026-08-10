from sentence_transformers import SentenceTransformer
from src.config import settings

_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model {settings.embedding_model} (downloads ~2GB on first run)...")
        _model = SentenceTransformer(settings.embedding_model)
        print("Model loaded.")
    return _model

def get_query_embedding(text: str) -> list[float]:
    return _get_model().encode("query: " + text).tolist()

def get_passage_embedding(text: str) -> list[float]:
    return _get_model().encode("passage: " + text).tolist()
