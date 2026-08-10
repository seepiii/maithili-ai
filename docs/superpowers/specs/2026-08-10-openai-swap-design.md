# Design: Replace Anthropic with OpenAI for chat generation

## Context

The backend currently calls the Anthropic API (`claude-sonnet-4-6`) in
`src/chat.py` to generate Maithili responses. The user has an OpenAI API key
but not Anthropic API billing, and wants to swap providers to unblock local
running and testing.

This is a provider swap only. It does not change the RAG + corrections
architecture: family corrections still enter ChromaDB at 10x priority and
steer retrieval; the LLM (now OpenAI) is only used at response-generation
time and does not need to be retrained or fine-tuned for corrections to
take effect.

## Approach

Fully replace Anthropic with OpenAI (no dual-provider config). Simplest
option, matches actual usage (one provider), no unused code paths.

- Model: `gpt-4o-mini` — cheapest capable OpenAI chat model, adequate for a
  RAG-assisted endpoint; can be swapped later by editing one string if
  quality is insufficient.

## Changes

- `src/chat.py` — replace `anthropic.Anthropic` client with `openai.OpenAI`.
  Swap `.messages.create(model="claude-sonnet-4-6", system=..., messages=[...])`
  for `.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", ...}, {"role": "user", ...}])`
  (OpenAI's chat format folds the system prompt into the messages list
  rather than a separate `system` param). Response text is read from
  `response.choices[0].message.content` instead of `message.content[0].text`.
- `src/config.py` — rename `anthropic_api_key` → `openai_api_key`.
- `.env.example` — `ANTHROPIC_API_KEY` → `OPENAI_API_KEY`.
- `pyproject.toml` — drop `anthropic` dependency, add `openai>=1.0.0`.
- `CLAUDE.md` — update "Stack" and "Key design decision" sections to
  reference OpenAI / gpt-4o-mini instead of Claude.

## Out of scope

- No dual-provider / configurable-provider support.
- No changes to embeddings (`multilingual-e5-large`, local, provider-agnostic),
  RAG retrieval, database, or other routers (`/correct`, `/contribute`,
  `/seed`, `/export/training`).
- No change to the corrections-priority mechanism.

## Testing

After implementation: run the server, hit `/docs`, and exercise `/chat`
with a real OpenAI key to confirm end-to-end generation works, plus
`/correct` and a follow-up `/chat` to confirm priority retrieval still
surfaces the correction as context.
