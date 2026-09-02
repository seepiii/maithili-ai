# Maithili AI Backend

FastAPI backend that makes an LLM fluent in Maithili (मैथिली) using RAG + family corrections.

## What this does

- `POST /chat` — submit a sentence, get its romanized Maithili translation from OpenAI
- `POST /correct` — family member submits a correction; it enters the vector store at 10x priority so it surfaces first in future responses
- `POST /contribute` — add Maithili text/phrases directly
- `POST /seed` — loads OPUS + Maithili Wikipedia into the vector store (run once)
- `GET /export/training` — downloads all corrections as a JSONL file (backup/export; not currently used for fine-tuning — see note below)

## Stack

- FastAPI + SQLite (conversations, corrections, contributions)
- ChromaDB at `data/chroma/` (local vector store, no cloud needed)
- OpenAI API for chat (`gpt-4o-mini`) and embeddings (`text-embedding-3-small`) — no local ML model, keeps the deploy footprint small

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # add your OPENAI_API_KEY
uvicorn src.main:app --reload
```

Then open http://localhost:8000/docs for the interactive UI.

### Terminal-only workflow (no server)

```bash
python3 chat_cli.py
```

Type a sentence, get a Maithili translation, type `c` to correct it. Every
translation prints what was retrieved from the vector store (source,
priority, similarity score) so you can see what actually influenced the
output. Same underlying logic as the API, no HTTP involved.

### Checking correction quality

```bash
python3 eval_corrections.py
```

Re-runs every corrected sentence through the live pipeline and reports two
things: whether each correction's own document is actually being retrieved
(and at what rank), and how similar the fresh LLM output is to the saved
correction. Run this periodically as the corpus grows to catch retrieval
regressions early (this is how a rank-#1 drop-off was caught and fixed once
already — see `rebuild_vector_store.py` below).

### Changing the embedding model

```bash
python3 rebuild_vector_store.py
```

If `EMBEDDING_MODEL` in `.env` changes, existing ChromaDB vectors (from the
old model) become incompatible — different models produce different
dimensions/semantic spaces. This script wipes the vector store and
re-embeds everything from SQLite (the source of truth for corrections and
contributions) using whatever model is currently configured.

## Key design decision

Corrections are stored with `priority=10.0` in ChromaDB. During retrieval, cosine similarity is multiplied by priority, so a family correction beats any corpus document for similar queries — no retraining needed.

**Note on fine-tuning:** the original design intended RAG as an interim step until enough corrections existed to fine-tune a real model via OpenAI's API. As of this writing, OpenAI has discontinued self-serve fine-tuning for new jobs (`403 training_not_available`), so RAG + corrections is the permanent mechanism here, not a stopgap. `GET /export/training` still exports the data in case a fine-tunable provider becomes available later.

## File layout

```
chat_cli.py           — terminal translate+correct loop, no server needed
eval_corrections.py   — measure retrieval rank + generation similarity for all corrections
rebuild_vector_store.py — re-embed all corrections/contributions after an EMBEDDING_MODEL change
src/
  main.py          — app entry point, inits SQLite on startup
  config.py        — settings from .env
  database.py      — SQLite (conversations, corrections, contributions)
  embeddings.py    — OpenAI embeddings wrapper
  vector_store.py  — ChromaDB insert + priority-weighted search
  rag.py           — embed query → search → format context
  chat.py          — OpenAI API + RAG context injection
  seed.py          — OPUS + Wikipedia seeding
  routers/
    chat.py        — POST /chat
    corrections.py — POST /correct, GET /corrections
    contributions.py — POST /contribute, GET /contributions
    admin.py       — POST /seed, GET /export/training
```
