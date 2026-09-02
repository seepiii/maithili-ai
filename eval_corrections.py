"""
Check whether corrections are actually taking effect.

There's no training loss to watch here — this is RAG, not fine-tuning
(OpenAI's fine-tuning platform is no longer available for new jobs, so RAG
is the permanent mechanism, not a stopgap). "Improving" means two separate
things you can actually measure:

1. Retrieval: for a corrected sentence, does its own correction come back
   as the top-ranked retrieved document? If not, the priority-boost search
   isn't finding it and the LLM never even sees it.
2. Generation: when re-asked, how closely does the fresh LLM output match
   the correction you saved? 100% match is rare and not required (RAG
   nudges, doesn't force output) — but this number should trend up as a
   sentence gets corrected multiple times, and stay reasonably high for
   sentences corrected once.

Usage:
    python3 eval_corrections.py
"""
from difflib import SequenceMatcher

from src.database import init_db, get_db
from src.chat import generate_response


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def main():
    init_db()
    with get_db() as db:
        rows = db.execute(
            """
            SELECT c.correct_response, conv.query
            FROM corrections c
            JOIN conversations conv ON conv.id = c.conversation_id
            ORDER BY c.created_at
            """
        ).fetchall()

    if not rows:
        print("No corrections in the database yet.")
        return

    sims = []
    retrieved_ranks = []

    print(f"Re-testing {len(rows)} corrected sentence(s)...\n")
    for row in rows:
        query, correct_response = row["query"], row["correct_response"]
        result = generate_response(query)
        fresh_output = result["response"]

        sim = similarity(fresh_output, correct_response)
        sims.append(sim)

        rank = next(
            (i + 1 for i, d in enumerate(result["retrieved_docs"]) if d["content"] == correct_response),
            None,
        )
        retrieved_ranks.append(rank)

        match_flag = "EXACT" if fresh_output.strip().lower() == correct_response.strip().lower() else ""
        print(f"Query:      {query}")
        print(f"Corrected:  {correct_response}")
        print(f"Fresh out:  {fresh_output}  {match_flag}")
        print(f"Similarity: {sim:.0%}   Retrieved rank: {rank if rank else 'NOT RETRIEVED'}")
        print()

    avg_sim = sum(sims) / len(sims)
    retrieved_ok = sum(1 for r in retrieved_ranks if r == 1)
    exact_matches = sum(1 for s in sims if s == 1.0)

    print("=" * 60)
    print(f"Average similarity to correction: {avg_sim:.0%}")
    print(f"Correction ranked #1 in retrieval: {retrieved_ok}/{len(rows)}")
    print(f"Exact matches: {exact_matches}/{len(rows)}")
    if retrieved_ok < len(rows):
        print("\nSome corrections aren't surfacing as the top retrieval result for")
        print("their own sentence — that's worth investigating (embedding model")
        print("quality, or too many similar corrections competing).")


if __name__ == "__main__":
    main()
