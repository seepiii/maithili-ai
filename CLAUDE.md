# Maithili AI Backend

FastAPI backend that makes Claude fluent in Maithili (मैथिली) using RAG + family corrections.

## What this does

- `POST /chat` — asks a question, gets a Maithili response from Claude
- `POST /correct` — family member submits a correction; it enters the vector store at 10x priority so it surfaces first in future responses
- `POST /contribute` — add Maithili text/phrases directly
- `POST /seed` — loads OPUS + Maithili Wikipedia into the vector store (run once)
- `GET /export/training` — downloads all corrections as a JSONL file for future fine-tuning

## Stack

- FastAPI + SQLite (conversations, corrections, contributions)
- ChromaDB at `data/chroma/` (local vector store, no cloud needed)
- `intfloat/multilingual-e5-large` for embeddings (downloads ~2GB on first use)
- Claude API (`claude-sonnet-4-6`) for chat

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # add your ANTHROPIC_API_KEY
uvicorn src.main:app --reload
```

Then open http://localhost:8000/docs for the interactive UI.

## Key design decision

Corrections are stored with `priority=10.0` in ChromaDB. During retrieval, cosine similarity is multiplied by priority, so a family correction beats any corpus document for similar queries — no retraining needed.

## File layout

```
src/
  main.py          — app entry point, inits SQLite on startup
  config.py        — settings from .env
  database.py      — SQLite (conversations, corrections, contributions)
  embeddings.py    — multilingual-e5-large wrapper (query/passage prefixes)
  vector_store.py  — ChromaDB insert + priority-weighted search
  rag.py           — embed query → search → format context
  chat.py          — Claude API + RAG context injection
  seed.py          — OPUS + Wikipedia seeding
  routers/
    chat.py        — POST /chat
    corrections.py — POST /correct, GET /corrections
    contributions.py — POST /contribute, GET /contributions
    admin.py       — POST /seed, GET /export/training
```
