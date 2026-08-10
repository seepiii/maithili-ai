import json
import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from src.chat import generate_response
from src.database import get_db

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat(req: ChatRequest):
    result = generate_response(req.message)
    conv_id = str(uuid.uuid4())

    with get_db() as db:
        db.execute(
            "INSERT INTO conversations (id, query, response, retrieved_doc_ids) VALUES (?, ?, ?, ?)",
            (conv_id, req.message, result["response"], json.dumps(result["retrieved_doc_ids"])),
        )

    return {
        "response": result["response"],
        "conversation_id": conv_id,
        "retrieved_doc_ids": result["retrieved_doc_ids"],
    }
