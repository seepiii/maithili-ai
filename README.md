# Maithili AI

A personal/family project that makes an LLM fluent in Maithili (मैथिली) —
the language spoken in the Mithila region of Bihar, India and Nepal — by
combining retrieval-augmented generation with corrections from actual
native speakers in the family.

**Live locally at:** `http://localhost:8000` (see [Run it](#run-it) — no
hosted deployment, this runs on your own machine)

## The idea

Generic LLMs are shaky at Maithili — it's a low-resource language, so
translations often come out generic or just wrong. Instead of trying to
fine-tune a model on a tiny dataset, this project takes a different
approach:

1. **Ask** — you type an English sentence, the AI (OpenAI `gpt-4o-mini`)
   translates it into romanized Maithili.
2. **Correct** — if a family member who actually speaks Maithili knows the
   translation is off, they fix it right there. The correction gets
   embedded and stored in a vector database at **10x priority**, so it
   outranks everything else for similar questions from then on.
3. **Contribute** — phrases and vocabulary can also be added directly,
   without needing a translation to correct first.
4. **Watch it learn** — every answer shows exactly what was retrieved to
   produce it (source, priority, similarity score), and a running
   training log shows every correction and contribution ever made, with
   an AI-generated explanation of *why* each correction is right (word
   choice, dialect, formality, grammar, etc).

No retraining, no fine-tuning job, no waiting — a correction influences
the very next relevant query.

**Note on fine-tuning:** the original plan was to use RAG + corrections
as an interim step until there was enough data to fine-tune a real model.
OpenAI has since discontinued self-serve fine-tuning for new jobs, so
RAG + corrections is the permanent mechanism here, not a stopgap.

## Stack

- **FastAPI** + **SQLite** — conversations, corrections, contributions
- **ChromaDB** (local, no cloud) — vector store for retrieval, at
  `data/chroma/`
- **`intfloat/multilingual-e5-base`** — multilingual embedding model
- **OpenAI API** (`gpt-4o-mini`) — translation + correction explanations
- Static HTML/CSS/JS frontend, no build step, served directly by FastAPI

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # add your OPENAI_API_KEY
uvicorn src.main:app --reload
```

Open **http://localhost:8000** for the web UI — translate a sentence,
correct it inline, add standalone phrases, and watch the training log
fill up. Or visit `/docs` for the raw interactive API.

### Terminal-only workflow (no server)

```bash
python3 chat_cli.py
```

Same translate → correct loop, no HTTP involved. Every translation
prints what was retrieved from the vector store so you can see what
actually influenced the output.

### Checking correction quality

```bash
python3 eval_corrections.py
```

Re-runs every corrected sentence through the live pipeline and reports
whether each correction's own document is actually being retrieved (and
at what rank), plus how similar the fresh LLM output is to the saved
correction. Useful for catching retrieval regressions as the corpus
grows.

### Changing the embedding model

```bash
python3 rebuild_vector_store.py
```

Wipes and re-embeds the vector store from SQLite (the source of truth)
after an `EMBEDDING_MODEL` change — old vectors from a different model
aren't compatible with a new one.

## How retrieval priority works

Corrections are stored with `priority=10.0` in ChromaDB. During
retrieval, cosine similarity is multiplied by priority, so a family
correction beats any corpus document for similar queries. That
multiplier is the entire "learning" mechanism — no model weights change,
just what gets surfaced into context.

## File layout

```
static/                — web UI (index.html, app.js, style.css)
chat_cli.py             — terminal translate+correct loop, no server needed
eval_corrections.py     — measure retrieval rank + generation similarity for all corrections
rebuild_vector_store.py — re-embed all corrections/contributions after an EMBEDDING_MODEL change
src/
  main.py          — app entry point, inits SQLite on startup, serves static/
  config.py        — settings from .env
  database.py      — SQLite (conversations, corrections, contributions)
  embeddings.py    — embedding model wrapper (query/passage prefixes)
  vector_store.py  — ChromaDB insert + priority-weighted search
  rag.py           — embed query → search → format context
  chat.py          — OpenAI API + RAG context injection
  seed.py          — OPUS + Wikipedia seeding
  routers/
    chat.py          — POST /chat
    corrections.py   — POST /correct, GET /corrections
    contributions.py — POST /contribute, GET /contributions
    admin.py         — POST /seed, GET /export/training
```
