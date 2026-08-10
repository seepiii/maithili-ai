import json
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from src.database import get_db
from src.seed import run_seed

router = APIRouter()

@router.post("/seed")
def trigger_seed(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_seed)
    return {"status": "seeding started — watch the terminal for progress"}

@router.get("/export/training")
def export_training_data():
    def generate():
        with get_db() as db:
            corrections = db.execute("SELECT * FROM corrections").fetchall()
            for c in corrections:
                conv = db.execute(
                    "SELECT query FROM conversations WHERE id = ?",
                    (c["conversation_id"],)
                ).fetchone()
                row = {
                    "prompt": conv["query"] if conv else "",
                    "chosen": c["correct_response"],
                    "rejected": c["wrong_response"],
                }
                yield json.dumps(row, ensure_ascii=False) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=maithili_training.jsonl"},
    )
