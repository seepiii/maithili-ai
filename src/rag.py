from src.embeddings import get_query_embedding
from src.vector_store import search_documents

def retrieve(query: str, limit: int = 10) -> dict:
    embedding = get_query_embedding(query)
    docs = search_documents(embedding, limit=limit)

    if not docs:
        return {"context": "", "doc_ids": [], "docs": []}

    context_parts = [
        f"[{doc['type'].upper()}] {doc['content']}"
        for doc in docs
    ]
    return {
        "context": "\n\n".join(context_parts),
        "doc_ids": [doc["id"] for doc in docs],
        "docs": docs,
    }
