import json
import uuid

from src.database import init_db, get_db
from src.chat import generate_response, explain_correction
from src.embeddings import get_passage_embedding
from src.vector_store import insert_document
from src.config import settings


def ask(query: str) -> str:
    result = generate_response(query)
    conv_id = str(uuid.uuid4())

    with get_db() as db:
        db.execute(
            "INSERT INTO conversations (id, query, response, retrieved_doc_ids) VALUES (?, ?, ?, ?)",
            (conv_id, query, result["response"], json.dumps(result["retrieved_doc_ids"])),
        )

    print(f"Maithili: {result['response']}")
    for doc in result["retrieved_docs"]:
        print(
            f"  [retrieved] {doc['type']:<10} priority={doc['priority']:<5} "
            f"score={doc['similarity']:.3f} source={doc['source']!r} :: {doc['content']}"
        )
    return conv_id


def correct(conv_id: str, correct_response: str, corrected_by: str = "family"):
    with get_db() as db:
        conv = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        corr_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO corrections (id, conversation_id, wrong_response, correct_response, corrected_by) VALUES (?, ?, ?, ?, ?)",
            (corr_id, conv_id, conv["response"], correct_response, corrected_by),
        )

    embedding = get_passage_embedding(correct_response)
    insert_document(
        content=correct_response,
        source=corrected_by,
        doc_type="correction",
        embedding=embedding,
        priority=settings.correction_priority,
        metadata={"conversation_id": conv_id, "correction_id": corr_id},
    )
    print("Correction saved — it'll surface first for similar questions from now on.")

    explanation = explain_correction(conv["response"], correct_response)
    print(f"Why: {explanation}")


def main():
    init_db()
    print("Maithili translator CLI. Type a sentence to translate, or 'quit' to exit.")
    print("After each translation you can type 'c' to correct it.\n")

    last_conv_id = None
    while True:
        query = input("Sentence: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        if query.lower() == "c":
            if not last_conv_id:
                print("Nothing to correct yet — translate a sentence first.")
                continue
            correct_response = input("Correct translation (romanized Maithili): ").strip()
            if correct_response:
                correct(last_conv_id, correct_response)
            continue

        last_conv_id = ask(query)


if __name__ == "__main__":
    main()
