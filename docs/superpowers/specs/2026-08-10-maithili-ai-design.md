# Maithili AI — Backend Design Spec
**Date:** 2026-08-10
**Status:** Approved

## Problem

Maithili (spoken in Bihar/Mithila region) has almost no high-quality AI resources. Existing models produce frequent errors — wrong vocabulary, wrong grammar, wrong register. This project builds a Maithili-aware chatbot backend where native speakers (initially one family) can correct the model's mistakes, and those corrections immediately improve future responses while building toward a fine-tuned model.

## Goal

A Python/FastAPI backend that:
1. Answers queries in Maithili using RAG over a seeded corpus + family contributions
2. Lets family members flag wrong responses and submit corrections
3. Weights corrections above generic corpus content in retrieval
4. Exports all corrections as fine-tuning training pairs for a future open-source Maithili model

Frontend is explicitly out of scope for this phase.

---

## Architecture

### Two-Phase Plan

**Phase 1 (now):** RAG chatbot powered by Claude API, backed by a vector store seeded with public Maithili corpora and enriched by family corrections.

**Phase 2 (later):** Export the correction dataset → fine-tune an open-source multilingual model (IndicBERT / LLaMA 3) → optionally replace Claude.

### Core Pipeline

```
Query arrives
    │
    ▼
Embed query (multilingual-e5-large)
    │
    ▼
Vector search (pgvector)
    ├── Corrections        ← retrieved first (10x priority weight)
    ├── Family contributions
    └── Seeded corpora     ← fallback
    │
    ▼
Claude API (claude-sonnet-4-6) + Maithili system prompt + retrieved context
    │
    ▼
Response returned
    │
[User flags as wrong]
    │
    ▼
Correction stored → re-embedded → enters retrieval pool immediately
Correction also saved as (wrong, correct, context) training pair
```

---

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| API framework | FastAPI (Python) | Fast, async, easy to extend |
| Database | Supabase (Postgres + pgvector) | One service: relational + vector search, free tier |
| LLM | Claude API `claude-sonnet-4-6` | Best multilingual reasoning |
| Embeddings | `intfloat/multilingual-e5-large` | Handles Devanagari/Maithili script |
| Corpus seeding | AI4Bharat, OPUS Maithili, Maithili Wikipedia | Best available public Maithili data |
| Training export | JSONL (Hugging Face format) | Ready for fine-tuning runs |

---

## API Endpoints

### Chat
```
POST /chat
Body: { "message": str, "session_id": str? }
Returns: { "response": str, "conversation_id": str, "retrieved_docs": [...] }
```

### Corrections
```
POST /correct
Body: { "conversation_id": str, "correct_response": str, "corrected_by": str }
Returns: { "correction_id": str }

GET /corrections
Returns: list of all corrections with metadata
```

### Contributions
```
POST /contribute
Body: {
  "text_maithili": str,
  "text_english": str?,
  "text_transliteration": str?,
  "contributor": str
}
Returns: { "contribution_id": str }
```

### Admin / Export
```
POST /seed              — trigger corpus seeding job
GET  /export/training   — export fine-tuning dataset as JSONL
```

---

## Data Models

### Document (vector store unit)
```
id            UUID
content       text           -- Maithili text
source        text           -- contributor name or corpus name
type          enum           -- correction | contribution | corpus
priority      float          -- default 1.0; corrections = 10.0
embedding     vector(1024)   -- multilingual-e5-large output
metadata      jsonb
created_at    timestamptz
```

### Conversation
```
id                UUID
query             text
response          text
retrieved_doc_ids UUID[]
created_at        timestamptz
```

### Correction
```
id               UUID
conversation_id  UUID  → Conversation
wrong_response   text
correct_response text
corrected_by     text
created_at       timestamptz
```

### Contribution
```
id                    UUID
text_maithili         text
text_english          text?
text_transliteration  text?
contributor           text
created_at            timestamptz
```

---

## Correction Priority Logic

Corrections are stored as Documents with `priority = 10.0`. During RAG retrieval, the vector similarity score is multiplied by `priority`, so corrections surface before corpus content for similar queries. This means a family correction takes effect on the next query — no retraining required.

---

## Seeding Strategy

On startup (or via `POST /seed`), the backend pulls from:
1. **AI4Bharat IndicNLP** — Maithili parallel sentences
2. **OPUS corpus** — Maithili-English sentence pairs
3. **Maithili Wikipedia** — article text (via Wikimedia API)

Each document is chunked (~200 tokens), embedded, and stored with `type=corpus` and `priority=1.0`.

---

## Fine-tuning Export Format

`GET /export/training` returns JSONL:
```json
{"prompt": "<original query>", "chosen": "<correct_response>", "rejected": "<wrong_response>"}
```
Compatible with Hugging Face TRL for DPO (Direct Preference Optimization) fine-tuning.

---

## Out of Scope (Phase 1)

- Frontend / UI (separate phase)
- Voice / speech input
- Authentication / user accounts
- Model hosting / inference server
- Public deployment
