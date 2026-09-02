"""
Rebuild the ChromaDB vector store from SQLite (the source of truth for
corrections/contributions) using the currently configured embedding model.

Needed whenever EMBEDDING_MODEL changes — embeddings from different models
aren't compatible (different dimensions/semantic space), so the old
collection has to be wiped and re-embedded, not merged into.

Usage:
    python3 rebuild_vector_store.py
"""
import shutil
import os

from src.database import init_db, get_db
from src.embeddings import get_passage_embedding
from src.vector_store import insert_document
from src.config import settings


def main():
    init_db()

    chroma_path = os.path.join(settings.data_dir, "chroma")
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print(f"Wiped old vector store at {chroma_path}")

    with get_db() as db:
        corrections = db.execute("SELECT * FROM corrections").fetchall()
        contributions = db.execute("SELECT * FROM contributions").fetchall()

    print(f"Re-embedding {len(corrections)} correction(s) with {settings.embedding_model}...")
    for c in corrections:
        embedding = get_passage_embedding(c["correct_response"])
        insert_document(
            content=c["correct_response"],
            source=c["corrected_by"],
            doc_type="correction",
            embedding=embedding,
            priority=settings.correction_priority,
            metadata={"conversation_id": c["conversation_id"], "correction_id": c["id"]},
        )

    print(f"Re-embedding {len(contributions)} contribution(s)...")
    for c in contributions:
        embedding = get_passage_embedding(c["text_maithili"])
        insert_document(
            content=c["text_maithili"],
            source=c["contributor"],
            doc_type="contribution",
            embedding=embedding,
            priority=1.0,
            metadata={"contribution_id": c["id"]},
        )

    print("Rebuild complete.")


if __name__ == "__main__":
    main()
