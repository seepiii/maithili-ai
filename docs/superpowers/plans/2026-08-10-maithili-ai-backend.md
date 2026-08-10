# Maithili AI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend that answers Maithili queries via RAG over a seeded corpus, lets family members submit corrections that immediately improve retrieval, and exports all corrections as DPO fine-tuning data.

**Architecture:** `multilingual-e5-large` embeds all text into Supabase pgvector. Corrections are stored with 10x priority weight so they surface before corpus content in retrieval. Claude API (`claude-sonnet-4-6`) generates responses using retrieved context. Every correction is saved as a `(prompt, chosen, rejected)` training triple in the `corrections` table.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, Supabase (Postgres + pgvector), Anthropic SDK, sentence-transformers (`intfloat/multilingual-e5-large`), httpx, datasets, pytest, pydantic-settings

---

## File Structure

```
maithili-ai/
├── src/
│   ├── __init__.py
│   ├── main.py              — FastAPI app, mounts all routers
│   ├── config.py            — pydantic-settings config from .env
│   ├── database.py          — Supabase client singleton
│   ├── embeddings.py        — multilingual-e5-large wrapper (query/passage prefixes)
│   ├── vector_store.py      — Document insert + weighted similarity search via RPC
│   ├── rag.py               — embed query → search → format context string
│   ├── chat.py              — Claude API handler, injects RAG context
│   ├── seed.py              — corpus seeding from OPUS + Maithili Wikipedia
│   └── routers/
│       ├── __init__.py
│       ├── chat.py          — POST /chat
│       ├── corrections.py   — POST /correct, GET /corrections
│       ├── contributions.py — POST /contribute
│       └── admin.py         — POST /seed, GET /export/training
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_rag.py
│   ├── test_chat.py
│   └── test_routers/
│       ├── __init__.py
│       ├── test_chat.py
│       ├── test_corrections.py
│       └── test_contributions.py
├── migrations/
│   └── 001_initial.sql
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "maithili-ai"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "supabase>=2.0.0",
    "anthropic>=0.40.0",
    "sentence-transformers>=3.0.0",
    "httpx>=0.27.0",
    "datasets>=2.20.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create .env.example**

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_MODEL=intfloat/multilingual-e5-large
CORRECTION_PRIORITY=10.0
```

- [ ] **Step 3: Create .gitignore**

```
.env
__pycache__/
*.pyc
.venv/
dist/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 4: Create src/__init__.py** (empty file)

- [ ] **Step 5: Create src/config.py**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    supabase_url: str
    supabase_key: str
    anthropic_api_key: str
    embedding_model: str = "intfloat/multilingual-e5-large"
    correction_priority: float = 10.0

    class Config:
        env_file = ".env"

settings = Settings()
```

- [ ] **Step 6: Create tests/__init__.py** (empty file)

- [ ] **Step 7: Create tests/conftest.py**

```python
from unittest.mock import MagicMock
import pytest
```

- [ ] **Step 8: Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: all packages install without error. `sentence-transformers` will download torch — this may take a few minutes on first run.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .env.example .gitignore src/__init__.py src/config.py tests/__init__.py tests/conftest.py
git commit -m "feat: project scaffolding and config"
```

---

## Task 2: Database schema + Supabase client

**Files:**
- Create: `migrations/001_initial.sql`
- Create: `src/database.py`

- [ ] **Step 1: Create migrations/001_initial.sql**

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Unified document store for corpus, contributions, and corrections
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('correction', 'contribution', 'corpus')),
    priority FLOAT NOT NULL DEFAULT 1.0,
    embedding vector(1024),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Log of every chat exchange (used to link corrections back to conversations)
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    retrieved_doc_ids UUID[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Family corrections: wrong→correct pairs, also stored in documents with high priority
CREATE TABLE IF NOT EXISTS corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    wrong_response TEXT NOT NULL,
    correct_response TEXT NOT NULL,
    corrected_by TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Direct text contributions from family members
CREATE TABLE IF NOT EXISTS contributions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text_maithili TEXT NOT NULL,
    text_english TEXT,
    text_transliteration TEXT,
    contributor TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Weighted similarity search: corrections (priority=10) surface above corpus (priority=1)
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1024),
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    source TEXT,
    type TEXT,
    priority FLOAT,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        d.id,
        d.content,
        d.source,
        d.type,
        d.priority,
        (1 - (d.embedding <=> query_embedding)) * d.priority AS similarity
    FROM documents d
    WHERE d.embedding IS NOT NULL
    ORDER BY similarity DESC
    LIMIT match_count;
$$;
```

- [ ] **Step 2: Run migration in Supabase**

1. Go to https://supabase.com → open your project → SQL Editor
2. Paste the full contents of `migrations/001_initial.sql`
3. Click Run

Expected: "Success. No rows returned."

If pgvector is not enabled, go to Database → Extensions → enable `vector`.

- [ ] **Step 3: Create src/database.py**

```python
from supabase import create_client, Client
from src.config import settings

_client: Client | None = None

def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client
```

- [ ] **Step 4: Verify connection (manual)**

```bash
source .venv/bin/activate
python -c "
from src.database import get_client
db = get_client()
result = db.table('documents').select('id').limit(1).execute()
print('Connected. Rows:', result.data)
"
```

Expected: `Connected. Rows: []`

- [ ] **Step 5: Commit**

```bash
git add migrations/001_initial.sql src/database.py
git commit -m "feat: database schema and supabase client"
```

---

## Task 3: Embedding service

**Files:**
- Create: `src/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_embeddings.py`:

```python
from unittest.mock import MagicMock, patch
import numpy as np

def test_get_query_embedding_returns_list_of_floats():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1] * 1024)

    with patch("src.embeddings._get_model", return_value=mock_model):
        from src.embeddings import get_query_embedding
        result = get_query_embedding("हमर नाम की अछि")

    assert isinstance(result, list)
    assert len(result) == 1024
    assert all(isinstance(v, float) for v in result)
    mock_model.encode.assert_called_once_with("query: हमर नाम की अछि")

def test_get_passage_embedding_uses_passage_prefix():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.2] * 1024)

    with patch("src.embeddings._get_model", return_value=mock_model):
        from src.embeddings import get_passage_embedding
        result = get_passage_embedding("माइथिली भाषा")

    mock_model.encode.assert_called_once_with("passage: माइथिली भाषा")
    assert len(result) == 1024
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_embeddings.py -v
```

Expected: `ImportError` — `src.embeddings` does not exist yet.

- [ ] **Step 3: Implement src/embeddings.py**

```python
from sentence_transformers import SentenceTransformer
from src.config import settings

_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model

def get_query_embedding(text: str) -> list[float]:
    return _get_model().encode("query: " + text).tolist()

def get_passage_embedding(text: str) -> list[float]:
    return _get_model().encode("passage: " + text).tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_embeddings.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/embeddings.py tests/test_embeddings.py
git commit -m "feat: multilingual-e5-large embedding service with query/passage prefixes"
```

---

## Task 4: Vector store operations

**Files:**
- Create: `src/vector_store.py`
- Create: `tests/test_vector_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vector_store.py`:

```python
from unittest.mock import MagicMock, patch
import pytest

@pytest.fixture
def mock_db():
    db = MagicMock()
    db.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "abc-123"}
    ]
    db.rpc.return_value.execute.return_value.data = [
        {
            "id": "doc-1",
            "content": "हामी माइथिली बाजै छी",
            "source": "family",
            "type": "contribution",
            "priority": 1.0,
            "similarity": 0.85,
        }
    ]
    return db

def test_insert_document_returns_id(mock_db):
    with patch("src.vector_store.get_client", return_value=mock_db):
        from src.vector_store import insert_document
        doc_id = insert_document(
            content="हामी माइथिली बाजै छी",
            source="family",
            doc_type="contribution",
            embedding=[0.1] * 1024,
            priority=1.0,
        )
    assert doc_id == "abc-123"

def test_search_documents_returns_matches(mock_db):
    with patch("src.vector_store.get_client", return_value=mock_db):
        from src.vector_store import search_documents
        results = search_documents(embedding=[0.1] * 1024, limit=5)
    assert len(results) == 1
    assert results[0]["content"] == "हामी माइथिली बाजै छी"
    assert results[0]["similarity"] == 0.85
    mock_db.rpc.assert_called_once_with(
        "match_documents",
        {"query_embedding": [0.1] * 1024, "match_count": 5},
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_vector_store.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/vector_store.py**

```python
from src.database import get_client

def insert_document(
    content: str,
    source: str,
    doc_type: str,
    embedding: list[float],
    priority: float = 1.0,
    metadata: dict | None = None,
) -> str:
    db = get_client()
    result = db.table("documents").insert({
        "content": content,
        "source": source,
        "type": doc_type,
        "embedding": embedding,
        "priority": priority,
        "metadata": metadata or {},
    }).execute()
    return result.data[0]["id"]

def search_documents(embedding: list[float], limit: int = 5) -> list[dict]:
    db = get_client()
    result = db.rpc("match_documents", {
        "query_embedding": embedding,
        "match_count": limit,
    }).execute()
    return result.data
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_vector_store.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/vector_store.py tests/test_vector_store.py
git commit -m "feat: vector store — insert documents and priority-weighted similarity search"
```

---

## Task 5: RAG retrieval pipeline

**Files:**
- Create: `src/rag.py`
- Create: `tests/test_rag.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rag.py`:

```python
from unittest.mock import patch

MOCK_DOCS = [
    {"id": "1", "content": "हामी माइथिली बाजै छी", "source": "family",
     "type": "contribution", "priority": 1.0, "similarity": 0.9},
    {"id": "2", "content": "राम गेला", "source": "corpus",
     "type": "corpus", "priority": 1.0, "similarity": 0.7},
]

def test_retrieve_returns_formatted_context():
    with patch("src.rag.get_query_embedding", return_value=[0.1] * 1024), \
         patch("src.rag.search_documents", return_value=MOCK_DOCS):
        from src.rag import retrieve
        result = retrieve("we speak maithili")

    assert "हामी माइथिली बाजै छी" in result["context"]
    assert "राम गेला" in result["context"]
    assert result["doc_ids"] == ["1", "2"]

def test_retrieve_empty_returns_empty_context():
    with patch("src.rag.get_query_embedding", return_value=[0.1] * 1024), \
         patch("src.rag.search_documents", return_value=[]):
        from src.rag import retrieve
        result = retrieve("no matches here")

    assert result["context"] == ""
    assert result["doc_ids"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_rag.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/rag.py**

```python
from src.embeddings import get_query_embedding
from src.vector_store import search_documents

def retrieve(query: str, limit: int = 5) -> dict:
    embedding = get_query_embedding(query)
    docs = search_documents(embedding, limit=limit)

    if not docs:
        return {"context": "", "doc_ids": []}

    context_parts = [
        f"[{doc['type'].upper()}] {doc['content']}"
        for doc in docs
    ]
    return {
        "context": "\n\n".join(context_parts),
        "doc_ids": [doc["id"] for doc in docs],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_rag.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/rag.py tests/test_rag.py
git commit -m "feat: RAG retrieval pipeline"
```

---

## Task 6: Claude chat handler

**Files:**
- Create: `src/chat.py`
- Create: `tests/test_chat.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat.py`:

```python
from unittest.mock import MagicMock, patch

def _mock_claude(response_text: str) -> MagicMock:
    content = MagicMock()
    content.text = response_text
    message = MagicMock()
    message.content = [content]
    client = MagicMock()
    client.messages.create.return_value = message
    return client

def test_generate_response_injects_context_into_prompt():
    mock_client = _mock_claude("हामी माइथिली बाजै छी।")

    with patch("src.chat._get_client", return_value=mock_client), \
         patch("src.chat.retrieve", return_value={
             "context": "हामी माइथिली बाजै छी",
             "doc_ids": ["doc-1"],
         }):
        from src.chat import generate_response
        result = generate_response("how do you say we speak maithili?")

    assert result["response"] == "हामी माइथिली बाजै छी।"
    assert result["retrieved_doc_ids"] == ["doc-1"]
    call_kwargs = mock_client.messages.create.call_args[1]
    assert "हामी माइथिली बाजै छी" in call_kwargs["messages"][0]["content"]

def test_generate_response_no_context_skips_context_block():
    mock_client = _mock_claude("माफ करू, मोहि नइ बुझाइत अछि।")

    with patch("src.chat._get_client", return_value=mock_client), \
         patch("src.chat.retrieve", return_value={"context": "", "doc_ids": []}):
        from src.chat import generate_response
        result = generate_response("xyz unknown")

    assert result["retrieved_doc_ids"] == []
    call_kwargs = mock_client.messages.create.call_args[1]
    assert "Context from native" not in call_kwargs["messages"][0]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chat.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/chat.py**

```python
import anthropic
from src.config import settings
from src.rag import retrieve

SYSTEM_PROMPT = """You are a fluent Maithili language assistant. Maithili (मैथिली) is spoken in the Mithila region of Bihar, India and Nepal.

Rules:
- Respond primarily in Maithili Devanagari script.
- If the user writes in English or Hindi, still respond in Maithili.
- The provided context comes from native speakers — treat it as ground truth. If it contradicts your training data, trust the context.
- If you are unsure of a Maithili word or phrase, say so rather than guessing.
- Common references: हामी (we/I informal), अछि (is/are), छी (are, polite), जाइत (going), आबैत (coming), की (what), कतए (where), माफ करू (excuse me/sorry), प्रणाम (respectful greeting)."""

_client: anthropic.Anthropic | None = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client

def generate_response(query: str) -> dict:
    retrieved = retrieve(query)
    context = retrieved["context"]

    if context:
        user_message = f"Context from native Maithili speakers:\n{context}\n\nUser question: {query}"
    else:
        user_message = query

    message = _get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return {
        "response": message.content[0].text,
        "retrieved_doc_ids": retrieved["doc_ids"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chat.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/chat.py tests/test_chat.py
git commit -m "feat: claude chat handler with RAG context injection"
```

---

## Task 7: Corrections router

**Files:**
- Create: `src/routers/__init__.py`
- Create: `src/routers/corrections.py`
- Create: `tests/test_routers/__init__.py`
- Create: `tests/test_routers/test_corrections.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_routers/__init__.py` (empty file).

Create `tests/test_routers/test_corrections.py`:

```python
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest

@pytest.fixture
def client():
    from src.routers.corrections import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

def test_post_correct_stores_correction_and_high_priority_document(client):
    mock_db = MagicMock()
    mock_db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {
        "id": "conv-1",
        "query": "how to greet",
        "response": "wrong response",
    }
    mock_db.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "corr-1"}
    ]

    with patch("src.routers.corrections.get_client", return_value=mock_db), \
         patch("src.routers.corrections.insert_document", return_value="doc-1") as mock_insert, \
         patch("src.routers.corrections.get_passage_embedding", return_value=[0.1] * 1024):
        response = client.post("/correct", json={
            "conversation_id": "conv-1",
            "correct_response": "नमस्कार / प्रणाम",
            "corrected_by": "Dadi",
        })

    assert response.status_code == 200
    assert response.json()["correction_id"] == "corr-1"
    call_kwargs = mock_insert.call_args[1]
    assert call_kwargs["doc_type"] == "correction"
    assert call_kwargs["priority"] == 10.0

def test_post_correct_404_for_unknown_conversation(client):
    mock_db = MagicMock()
    mock_db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = None

    with patch("src.routers.corrections.get_client", return_value=mock_db):
        response = client.post("/correct", json={
            "conversation_id": "bad-id",
            "correct_response": "anything",
            "corrected_by": "Dadi",
        })

    assert response.status_code == 404

def test_get_corrections_returns_list(client):
    mock_db = MagicMock()
    mock_db.table.return_value.select.return_value.order.return_value.execute.return_value.data = [
        {"id": "corr-1", "correct_response": "नमस्कार", "corrected_by": "Dadi",
         "conversation_id": "conv-1", "wrong_response": "wrong", "created_at": "2026-01-01"}
    ]

    with patch("src.routers.corrections.get_client", return_value=mock_db):
        response = client.get("/corrections")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["corrected_by"] == "Dadi"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_routers/test_corrections.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create src/routers/__init__.py** (empty file)

- [ ] **Step 4: Implement src/routers/corrections.py**

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.database import get_client
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
    db = get_client()

    conv = db.table("conversations").select("*").eq(
        "id", req.conversation_id
    ).single().execute().data
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    corr = db.table("corrections").insert({
        "conversation_id": req.conversation_id,
        "wrong_response": conv["response"],
        "correct_response": req.correct_response,
        "corrected_by": req.corrected_by,
    }).execute()
    correction_id = corr.data[0]["id"]

    embedding = get_passage_embedding(req.correct_response)
    insert_document(
        content=req.correct_response,
        source=req.corrected_by,
        doc_type="correction",
        embedding=embedding,
        priority=settings.correction_priority,
        metadata={"conversation_id": req.conversation_id, "correction_id": correction_id},
    )

    return {"correction_id": correction_id}

@router.get("/corrections")
def list_corrections():
    db = get_client()
    result = db.table("corrections").select("*").order("created_at", desc=True).execute()
    return result.data
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_routers/test_corrections.py -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/routers/__init__.py src/routers/corrections.py tests/test_routers/__init__.py tests/test_routers/test_corrections.py
git commit -m "feat: corrections router — corrections stored as 10x priority documents"
```

---

## Task 8: Contributions router

**Files:**
- Create: `src/routers/contributions.py`
- Create: `tests/test_routers/test_contributions.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_routers/test_contributions.py`:

```python
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest

@pytest.fixture
def client():
    from src.routers.contributions import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

def test_post_contribute_stores_contribution_and_document(client):
    mock_db = MagicMock()
    mock_db.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "contrib-1"}
    ]

    with patch("src.routers.contributions.get_client", return_value=mock_db), \
         patch("src.routers.contributions.insert_document", return_value="doc-1"), \
         patch("src.routers.contributions.get_passage_embedding", return_value=[0.1] * 1024):
        response = client.post("/contribute", json={
            "text_maithili": "हामी सब मिलि कऽ काज करैत छी",
            "text_english": "We all work together",
            "contributor": "Nana",
        })

    assert response.status_code == 200
    assert response.json()["contribution_id"] == "contrib-1"

def test_post_contribute_text_maithili_is_required(client):
    response = client.post("/contribute", json={
        "text_english": "missing maithili text",
        "contributor": "Nana",
    })
    assert response.status_code == 422

def test_post_contribute_english_is_optional(client):
    mock_db = MagicMock()
    mock_db.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "contrib-2"}
    ]

    with patch("src.routers.contributions.get_client", return_value=mock_db), \
         patch("src.routers.contributions.insert_document", return_value="doc-2"), \
         patch("src.routers.contributions.get_passage_embedding", return_value=[0.1] * 1024):
        response = client.post("/contribute", json={
            "text_maithili": "राम गेला",
            "contributor": "Baba",
        })

    assert response.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_routers/test_contributions.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/routers/contributions.py**

```python
from fastapi import APIRouter
from pydantic import BaseModel
from src.database import get_client
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
    db = get_client()

    contrib = db.table("contributions").insert({
        "text_maithili": req.text_maithili,
        "text_english": req.text_english,
        "text_transliteration": req.text_transliteration,
        "contributor": req.contributor,
    }).execute()
    contribution_id = contrib.data[0]["id"]

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
        metadata={"contribution_id": contribution_id},
    )

    return {"contribution_id": contribution_id}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_routers/test_contributions.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/routers/contributions.py tests/test_routers/test_contributions.py
git commit -m "feat: contributions router"
```

---

## Task 9: Chat router + conversation logging

**Files:**
- Create: `src/routers/chat.py`
- Create: `tests/test_routers/test_chat.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_routers/test_chat.py`:

```python
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest

@pytest.fixture
def client():
    from src.routers.chat import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

def test_post_chat_returns_response_and_conversation_id(client):
    mock_db = MagicMock()
    mock_db.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "conv-123"}
    ]

    with patch("src.routers.chat.generate_response", return_value={
        "response": "हामी माइथिली बाजै छी।",
        "retrieved_doc_ids": ["doc-1"],
    }), patch("src.routers.chat.get_client", return_value=mock_db):
        response = client.post("/chat", json={"message": "how do you say we speak Maithili"})

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "हामी माइथिली बाजै छी।"
    assert data["conversation_id"] == "conv-123"
    assert data["retrieved_doc_ids"] == ["doc-1"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_routers/test_chat.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/routers/chat.py**

```python
from fastapi import APIRouter
from pydantic import BaseModel
from src.chat import generate_response
from src.database import get_client

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat(req: ChatRequest):
    result = generate_response(req.message)

    db = get_client()
    conv = db.table("conversations").insert({
        "query": req.message,
        "response": result["response"],
        "retrieved_doc_ids": result["retrieved_doc_ids"],
    }).execute()

    return {
        "response": result["response"],
        "conversation_id": conv.data[0]["id"],
        "retrieved_doc_ids": result["retrieved_doc_ids"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_routers/test_chat.py -v
```

Expected: 1 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/routers/chat.py tests/test_routers/test_chat.py
git commit -m "feat: chat router with conversation logging"
```

---

## Task 10: Corpus seeding

**Files:**
- Create: `src/seed.py`

- [ ] **Step 1: Implement src/seed.py**

```python
import httpx
from src.vector_store import insert_document
from src.embeddings import get_passage_embedding

def _chunk_text(text: str, max_chars: int = 400) -> list[str]:
    sentences = text.split("।")
    chunks, current = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) > max_chars and current:
            chunks.append(current.strip())
            current = s
        else:
            current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks

def _seed_opus(limit: int = 500) -> int:
    try:
        from datasets import load_dataset
        ds = load_dataset("opus100", "en-mai", split="train", streaming=True)
        count = 0
        for example in ds:
            if count >= limit:
                break
            mai = example.get("translation", {}).get("mai", "")
            en = example.get("translation", {}).get("en", "")
            if not mai:
                continue
            content = mai + (f" ({en})" if en else "")
            embedding = get_passage_embedding(content)
            insert_document(
                content=content,
                source="opus100",
                doc_type="corpus",
                embedding=embedding,
                priority=1.0,
                metadata={"en": en},
            )
            count += 1
        return count
    except Exception as e:
        print(f"OPUS seeding error: {e}")
        return 0

def _seed_wikipedia(limit: int = 300) -> int:
    api = "https://mai.wikipedia.org/w/api.php"
    count = 0
    try:
        pages = httpx.get(api, params={
            "action": "query", "list": "random",
            "rnnamespace": 0, "rnlimit": 20, "format": "json",
        }, timeout=30).json().get("query", {}).get("random", [])

        for page in pages:
            if count >= limit:
                break
            extract = httpx.get(api, params={
                "action": "query", "pageids": page["id"],
                "prop": "extracts", "exintro": True,
                "explaintext": True, "format": "json",
            }, timeout=30).json().get("query", {}).get("pages", {}).get(
                str(page["id"]), {}
            ).get("extract", "")

            if not extract or len(extract) < 50:
                continue

            for chunk in _chunk_text(extract):
                if count >= limit:
                    break
                embedding = get_passage_embedding(chunk)
                insert_document(
                    content=chunk,
                    source="mai.wikipedia.org",
                    doc_type="corpus",
                    embedding=embedding,
                    priority=1.0,
                    metadata={"page_id": page["id"]},
                )
                count += 1
        return count
    except Exception as e:
        print(f"Wikipedia seeding error: {e}")
        return 0

def run_seed():
    print("Seeding OPUS Maithili corpus...")
    n1 = _seed_opus(limit=500)
    print(f"  Seeded {n1} OPUS sentences")

    print("Seeding Maithili Wikipedia...")
    n2 = _seed_wikipedia(limit=300)
    print(f"  Seeded {n2} Wikipedia chunks")

    print(f"Seeding complete. Total: {n1 + n2} documents")
```

- [ ] **Step 2: Commit**

```bash
git add src/seed.py
git commit -m "feat: corpus seeding from OPUS en-mai and Maithili Wikipedia"
```

---

## Task 11: Admin router (seed trigger + training export)

**Files:**
- Create: `src/routers/admin.py`

- [ ] **Step 1: Implement src/routers/admin.py**

```python
import json
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from src.database import get_client
from src.seed import run_seed

router = APIRouter()

@router.post("/seed")
def trigger_seed(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_seed)
    return {"status": "seeding started in background"}

@router.get("/export/training")
def export_training_data():
    db = get_client()
    corrections = db.table("corrections").select("*").execute().data

    def generate():
        for c in corrections:
            conv = db.table("conversations").select("query").eq(
                "id", c["conversation_id"]
            ).single().execute().data
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
```

- [ ] **Step 2: Commit**

```bash
git add src/routers/admin.py
git commit -m "feat: admin router — corpus seed trigger and DPO training export"
```

---

## Task 12: Main app + run the server

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Run all tests to confirm everything passes**

```bash
pytest tests/ -v
```

Expected: all tests pass. Fix any failures before continuing.

- [ ] **Step 2: Create src/main.py**

```python
from fastapi import FastAPI
from src.routers import chat, corrections, contributions, admin

app = FastAPI(title="Maithili AI Backend", version="0.1.0")

app.include_router(chat.router)
app.include_router(corrections.router)
app.include_router(contributions.router)
app.include_router(admin.router)
```

- [ ] **Step 3: Copy .env and fill in credentials**

```bash
cp .env.example .env
```

Edit `.env` with:
- `SUPABASE_URL` — from Supabase project Settings → API → Project URL
- `SUPABASE_KEY` — from Settings → API → `anon` `public` key
- `ANTHROPIC_API_KEY` — from console.anthropic.com → API Keys

- [ ] **Step 4: Start the server**

```bash
source .venv/bin/activate
uvicorn src.main:app --reload
```

Expected:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [...]
```

- [ ] **Step 5: Open the interactive API docs**

Open http://localhost:8000/docs in a browser. You will see all 6 endpoints with forms to test them — no curl needed.

- [ ] **Step 6: Test — send a chat message**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do you say hello in Maithili?"}'
```

Copy the `conversation_id` from the response — you need it for the next step.

Expected response shape:
```json
{
  "response": "प्रणाम / नमस्कार ...",
  "conversation_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "retrieved_doc_ids": []
}
```

- [ ] **Step 7: Test — seed the corpus (runs in background)**

```bash
curl -X POST http://localhost:8000/seed
```

Expected: `{"status": "seeding started in background"}`

Watch the server terminal — it will print seeding progress. After seeding, `/chat` responses will include retrieved Maithili context.

- [ ] **Step 8: Test — submit a family correction**

Replace `<conversation_id>` with the id from Step 6:

```bash
curl -X POST http://localhost:8000/correct \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "<conversation_id>",
    "correct_response": "प्रणाम — ई माइथिली में शुभकामना थिक",
    "corrected_by": "Dadi"
  }'
```

Expected: `{"correction_id": "..."}`

Send the same chat message again — the correction will now appear in retrieved context.

- [ ] **Step 9: Test — add a direct family contribution**

```bash
curl -X POST http://localhost:8000/contribute \
  -H "Content-Type: application/json" \
  -d '{
    "text_maithili": "हामी सब मिलि कऽ खाना खाइत छी",
    "text_english": "We all eat together",
    "contributor": "Nana"
  }'
```

Expected: `{"contribution_id": "..."}`

- [ ] **Step 10: Test — export training data**

```bash
curl http://localhost:8000/export/training -o training.jsonl
cat training.jsonl
```

Expected: one JSONL line per correction, each with `prompt`, `chosen`, `rejected` fields.

- [ ] **Step 11: Commit**

```bash
git add src/main.py
git commit -m "feat: main app entry point — maithili AI backend complete"
```
