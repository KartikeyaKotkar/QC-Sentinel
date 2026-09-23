from __future__ import annotations

import logging

import chromadb

from src.config import settings
from src.schemas import BugReport

logger = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_CHARS_FALLBACK = 2000

_model = None  # lazy SentenceTransformer


def get_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers not installed") from exc
    _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    try:
        _model.encode(["warmup"], normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        pass
    logger.info("Loaded embedding model %s", settings.EMBEDDING_MODEL)
    return _model


def get_collection():
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def build_document_text(bug: BugReport) -> str:
    text = f"{bug.title}\n\n{bug.description or ''}".strip()
    try:
        model = get_model()
        tokenizer = model.tokenizer  # type: ignore[attr-defined]
        tokens = tokenizer.encode(text, truncation=False)  # type: ignore
        if len(tokens) > 512:
            tokens = tokens[:512]
            text = tokenizer.decode(tokens, skip_special_tokens=True)  # type: ignore
    except Exception:
        text = text[:MAX_CHARS_FALLBACK]
    return text


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    return embeddings.tolist()  # type: ignore


def ingest_bugs(bugs: list[BugReport]) -> int:
    if not bugs:
        return 0
    collection = get_collection()
    texts = [build_document_text(b) for b in bugs]
    embeddings = embed_texts(texts)
    ids = [b.bug_id for b in bugs]  # type: ignore
    documents = texts
    metadatas = [
        {"subsystem": (b.subsystem or "Core").strip().lower(), "bug_id": b.bug_id, "state": b.state or "closed"}  # type: ignore
        for b in bugs
    ]
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],  # type: ignore
            embeddings=embeddings[i : i + batch_size],
        )
    logger.info("Upserted %d bugs into %s", len(ids), settings.CHROMA_COLLECTION)
    return len(ids)


def load_raw_and_ingest(path: str = "data/raw/godot_qc_bugs.json") -> int:
    import json

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    bugs = [BugReport(**item) for item in raw]
    return ingest_bugs(bugs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest bugs into ChromaDB")
    parser.add_argument("--input", type=str, default="data/raw/godot_qc_bugs.json")
    args = parser.parse_args()
    count = load_raw_and_ingest(args.input)
    print(f"Ingested {count} bugs from {args.input}")
