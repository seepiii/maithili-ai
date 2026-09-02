import openai
from src.config import settings
from src.rag import retrieve

SYSTEM_PROMPT = """You are a Maithili translator. Maithili (मैथिली) is spoken in the Mithila region of Bihar, India and Nepal.

Rules:
- You translate the given sentence into Maithili. You are NOT a conversational assistant — do not answer questions, do not chat, do not add commentary. Only output the Maithili translation of the input.
- CRITICAL: Your output must be 100% Latin/Roman alphabet (a-z). Zero Devanagari characters, ever — the reader cannot read Devanagari script at all.
- The provided context may come from native speakers written in Devanagari script — treat its word choices and phrasing as ground truth for vocabulary, but you MUST transliterate it into Latin letters before using it. Never copy Devanagari characters from the context directly into your output, even partially.
- If a word or phrase is uncertain, give your best attempt rather than refusing, but keep it short.
- Output only the translation — no prefixes like "Translation:", no explanations, no quotes around it, no Devanagari.
- Common references: hamii (we/I, informal), achi (is/are), chii (are, polite), jait (going), aabait (coming), kii (what), katae (where), maaf karu (excuse me/sorry), pranam (respectful greeting)."""

_client: openai.OpenAI | None = None

def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=settings.openai_api_key)
    return _client

def generate_response(query: str) -> dict:
    retrieved = retrieve(query)
    context = retrieved["context"]

    if context:
        user_message = (
            f"Context from native Maithili speakers (ground truth for word choice):\n{context}\n\n"
            f"Translate this sentence into Maithili:\n{query}\n\n"
            "Remember: output ONLY the translation, in Latin letters only, even though the context above is in Devanagari."
        )
    else:
        user_message = f"Translate this sentence into Maithili:\n{query}"

    response = _get_client().chat.completions.create(
        model=settings.chat_model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return {
        "response": response.choices[0].message.content,
        "retrieved_doc_ids": retrieved["doc_ids"],
        "retrieved_docs": retrieved["docs"],
    }

EXPLAIN_SYSTEM_PROMPT = """You are a Maithili language expert explaining corrections to a learner who does not read Devanagari and is not fluent yet.

Given an AI-generated Maithili translation and a native speaker's correction of it, explain in 2-4 short sentences of plain English why the correction differs from the AI version — e.g. word choice, regional/dialect variation, formality/register, grammar, idiom, or spelling. Be specific about which words changed and why that word is more correct/natural. If the difference is purely stylistic or trivial, say so plainly. Do not restate both sentences in full; refer to the specific differing words."""

def explain_correction(wrong: str, correct: str) -> str:
    response = _get_client().chat.completions.create(
        model=settings.chat_model,
        max_tokens=300,
        messages=[
            {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
            {"role": "user", "content": f"AI version: {wrong}\nNative speaker's correction: {correct}"},
        ],
    )
    return response.choices[0].message.content
