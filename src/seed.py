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
                    metadata={"page_id": str(page["id"])},
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
