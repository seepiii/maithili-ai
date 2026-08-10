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
