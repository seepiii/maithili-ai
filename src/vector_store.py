import os
import chromadb
from src.config import settings

_collection = None

def _get_collection():
    global _collection
    if _collection is None:
        chroma_path = os.path.join(settings.data_dir, "chroma")
        os.makedirs(chroma_path, exist_ok=True)
        client = chromadb.PersistentClient(path=chroma_path)
        _collection = client.get_or_create_collection(
            "maithili",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def insert_document(
    content: str,
    source: str,
    doc_type: str,
    embedding: list[float],
    priority: float = 1.0,
    metadata: dict | None = None,
) -> str:
    import uuid
    doc_id = str(uuid.uuid4())
    meta = {"source": source, "type": doc_type, "priority": priority}
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})

    _get_collection().add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[meta],
    )
    return doc_id

def search_documents(embedding: list[float], limit: int = 5) -> list[dict]:
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return []

    # Fetch extra so priority re-ranking has enough candidates
    n = min(total, max(limit * 4, 20))
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    scored = []
    for doc_id, content, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        priority = float(meta.get("priority", 1.0))
        scored.append({
            "id": doc_id,
            "content": content,
            "source": meta.get("source", ""),
            "type": meta.get("type", "corpus"),
            "priority": priority,
            "similarity": (1 - dist) * priority,
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:limit]
