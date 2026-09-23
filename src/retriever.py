from __future__ import annotations

import logging

from src.config import settings
from src.ingest import BGE_QUERY_PREFIX, MAX_CHARS_FALLBACK, get_collection, get_model
from src.schemas import RetrievedBug

logger = logging.getLogger(__name__)


def _truncate_query(text: str) -> str:
    text = text.strip()
    if not text:
        return text
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


def query_bugs(
    query_text: str,
    subsystem: str | None = None,
    top_k: int | None = None,
) -> list[RetrievedBug]:
    if not query_text or not query_text.strip():
        return []

    text = _truncate_query(query_text)
    k = top_k or settings.TOP_K

    where = None
    if subsystem and subsystem != "All":
        # DB stores lowercase (e.g. "rendering"), query may be "Rendering" — normalize
        where = {"subsystem": subsystem.strip().lower()}

    try:
        collection = get_collection()
        try:
            if collection.count() == 0:
                return []
        except Exception:
            pass

        model = get_model()
        embeddings = model.encode(
            [BGE_QUERY_PREFIX + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        results = collection.query(
            query_embeddings=embeddings.tolist(),  # type: ignore
            n_results=k,
            where=where,
        )
    except Exception as exc:
        logger.error("Chroma query failed: %s", exc)
        return []

    ids = (results.get("ids") or [[]])[0]
    if not ids:
        return []

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    out: list[RetrievedBug] = []
    for i, bug_id in enumerate(ids):
        doc = documents[i] if i < len(documents) and documents[i] is not None else ""
        meta = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
        dist = float(distances[i]) if i < len(distances) and distances[i] is not None else 0.0

        if "\n\n" in doc:
            title, description = doc.split("\n\n", 1)
        else:
            title, description = doc, ""

        out.append(
            RetrievedBug(
                bug_id=str(meta.get("bug_id") or bug_id),
                title=title.strip() or str(bug_id),
                description=description.strip(),
                subsystem=str(meta.get("subsystem") or "Core"),
                distance=dist,
                metadata=dict(meta),
            )
        )

    return out


search_with_scores = query_bugs
