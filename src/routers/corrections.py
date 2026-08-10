import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.database import get_db
from src.vector_store import insert_document
from src.embeddings import get_passage_embedding
from src.config import settings

router = APIRouter()

class CorrectRequest(BaseModel):
    conversation_id: str
    correct_response: str
    corrected_by: str

@router.post("/correct")
def submit_correction(req: CorrectRequest):
    with get_db() as db:
        conv = db.execute(
            "SELECT * FROM conversations WHERE id = ?", (req.conversation_id,)
        ).fetchone()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        corr_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO corrections (id, conversation_id, wrong_response, correct_response, corrected_by) VALUES (?, ?, ?, ?, ?)",
            (corr_id, req.conversation_id, conv["response"], req.correct_response, req.corrected_by),
        )

    embedding = get_passage_embedding(req.correct_response)
    insert_document(
        content=req.correct_response,
        source=req.corrected_by,
        doc_type="correction",
        embedding=embedding,
        priority=settings.correction_priority,
        metadata={"conversation_id": req.conversation_id, "correction_id": corr_id},
    )

    return {"correction_id": corr_id}

@router.get("/corrections")
def list_corrections():
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM corrections ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
