import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from src.database import get_db
from src.vector_store import insert_document
from src.embeddings import get_passage_embedding

router = APIRouter()

class ContributeRequest(BaseModel):
    text_maithili: str
    text_english: str | None = None
    text_transliteration: str | None = None
    contributor: str

@router.post("/contribute")
def add_contribution(req: ContributeRequest):
    contrib_id = str(uuid.uuid4())

    with get_db() as db:
        db.execute(
            "INSERT INTO contributions (id, text_maithili, text_english, text_transliteration, contributor) VALUES (?, ?, ?, ?, ?)",
            (contrib_id, req.text_maithili, req.text_english, req.text_transliteration, req.contributor),
        )

    content = req.text_maithili
    if req.text_english:
        content += f" ({req.text_english})"

    embedding = get_passage_embedding(content)
    insert_document(
        content=content,
        source=req.contributor,
        doc_type="contribution",
        embedding=embedding,
        priority=1.0,
        metadata={"contribution_id": contrib_id},
    )

    return {"contribution_id": contrib_id}

@router.get("/contributions")
def list_contributions():
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM contributions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
