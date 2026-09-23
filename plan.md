# QC-Sentinel: RAG-Powered Game Engine Bug Triage Engine

An automated Quality Control (QC) and testing triage tool that ingests game engine crash logs, detects duplicate defect reports via dense semantic retrieval, and produces root-cause hypotheses and reproduction insights using local-first LLM inference.

---

## 1. System Architecture

```
[Incoming Bug / Crash Report]
│
▼
[FastAPI Endpoint (/api/v1/triage)]  ── Pydantic v2 validation + normalization
│
▼
[Query Preprocessing & Metadata Filter]
│  • BGE query instruction prefix: "Represent this sentence for searching relevant passages: "
│  • Text normalization: title + "\n\n" + description, truncated to 512 tokens (BGE limit)
│  • Optional where-filter on `subsystem` (exact match)
│
▼
[ChromaDB Dense Vector Retrieval]
│  • Model: BAAI/bge-base-en-v1.5 (768-dim, normalize_embeddings=True)
│  • Collection: `qc_bugs` | Distance: cosine | HNSW
│  • Output: Top-k (k=5 default) with distances + metadata
│  • Threshold gate: if min_distance > DISTANCE_THRESHOLD → skip LLM, return non-duplicate
│
▼
[LLM Reasoning & Triage Layer]
   • Engine: LiteLLM → primary `ollama/llama3` (local), fallback configurable via LLM_PROVIDER
   • Prompt: System = "Game Engine QC Specialist" + retrieved context (top-k)
   • Schema Enforcement: Pydantic v2 + JSON mode with retry on parse failure
│
▼
[Structured Triage Output (JSON)]
   • is_duplicate: bool
   • duplicate_of: str | null  (bug_id)
   • confidence_score: float 0.0-1.0
   • root_cause_hypothesis: str
   • suggested_workaround: str
   • retrieved_context: list[RetrievedBug] (for explainability)
```

**Design decisions:**
- **Dense-only v1, hybrid later:** BM25+dense reranking is a v2 optimization. v1 proves dense retrieval quality; keep scope tight.
- **Local-first with fallback:** `LLM_PROVIDER=ollama` by default. `groq/llama-3.1-8b-instant` is opt-in via env, not equivalent — documented as fallback for CI/demo where Ollama RAM is constrained.
- **Threshold gate saves cost:** Vector distance > threshold → deterministic `is_duplicate=false` without LLM call. Threshold tuned in Phase 2 eval, not hardcoded.

---

## 2. Directory Structure

```text
qc-sentinel/
├── data/
│   ├── raw/                  # Scraped GitHub issues (godot_qc_bugs.json) + synthetic logs
│   └── chroma_db/            # Chroma PersistentClient storage (Docker named volume)
├── scripts/
│   └── fetch_godot_issues.py # GitHub API scraper (handles pagination, rate limit, PR filter)
├── src/
│   ├── __init__.py
│   ├── config.py             # Pydantic Settings (env vars, see §3.1)
│   ├── schemas.py            # Request/Response DTOs (Pydantic v2)
│   ├── ingest.py             # Batch embedding + upsert pipeline
│   ├── retriever.py          # Vector search + metadata filtering
│   ├── triage.py             # Prompt builder + LiteLLM orchestration + JSON repair
│   └── main.py               # FastAPI app, routes, middleware, exception handlers
├── tests/
│   ├── test_ingest.py        # Upsert idempotency, chunking, empty body handling
│   ├── test_retriever.py     # Cosine correctness, subsystem filter, latency <250ms
│   └── test_api.py           # /health, /triage, /ingest, 422/500 paths
├── eval/
│   ├── benchmark_dataset.json # Labeled duplicate pairs (schema §5)
│   └── evaluate_retrieval.py # HitRate@1/3/5, MRR, p50/p95 latency
├── .env.example
├── .gitignore
├── Dockerfile                # Single-stage Python; Ollama is separate compose service
├── docker-compose.yml        # api + ollama + chroma volume
├── requirements.txt
└── README.md
```

---

## 3. Configuration

### 3.1 `src/config.py` — Pydantic Settings

| Variable | Default | Description |
|---|---|---|
| `CHROMA_PATH` | `./data/chroma_db` | PersistentClient path |
| `CHROMA_COLLECTION` | `qc_bugs` | Collection name |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | SentenceTransformers model |
| `TOP_K` | `5` | Retrieval k |
| `DISTANCE_THRESHOLD` | `0.35` | Cosine distance gate (tuned via eval) |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama base URL (compose service name) |
| `LLM_PROVIDER` | `ollama` | `ollama` or `groq` |
| `LLM_MODEL` | `ollama/llama3` | LiteLLM model string |
| `LLM_TIMEOUT` | `15` | Seconds |
| `LLM_MAX_RETRIES` | `2` | LiteLLM retries on timeout/parse fail |
| `API_KEY` | `null` | If set, `POST /api/v1/ingest` requires `X-API-Key` |
| `GITHUB_TOKEN` | `null` | For `fetch_godot_issues.py` (5000 req/hr vs 60) |

All loaded from `.env` via `pydantic-settings`.

### 3.2 `requirements.txt` (pinned)

```text
fastapi==0.110.2
uvicorn[standard]==0.29.0
pydantic==2.7.1
pydantic-settings==2.2.1
chromadb==0.5.5
sentence-transformers==2.6.1
litellm==1.40.0
requests==2.32.0
pytest==8.2.0
httpx==0.27.0
python-multipart==0.0.9
structlog==24.1.0
onnxruntime==1.18.0  # optional, for CPU embedding speedup
```

> Note: `chromadb>=0.4.24` had breaking `PersistentClient` changes; pin to `0.5.5` (current stable) and use `chromadb.PersistentClient(path=CHROMA_PATH)`.

### 3.3 `src/schemas.py` — Key Schemas

```python
class BugReport(BaseModel):
    bug_id: str | None = None  # auto-generated if missing (TRIAGE-xxx)
    title: str = Field(..., min_length=1, max_length=500)
    description: str | None = None  # normalized to "" if null; never 500
    subsystem: str | None = "Core"  # normalized to "Core" if null/empty
    labels: list[str] = []

    @field_validator("description", mode="before")
    def normalize_desc(cls, v): return v or ""

class TriageResponse(BaseModel):
    is_duplicate: bool
    duplicate_of: str | None
    confidence_score: float = Field(ge=0.0, le=1.0)
    root_cause_hypothesis: str
    suggested_workaround: str
    retrieved_context: list[RetrievedBug]  # top-k with distances for explainability

class IngestPayload(BaseModel):
    bugs: list[BugReport]
```

- `BugReport` validators ensure malformed stack traces / missing `subsystem` → normalized, not 500. API returns `422` only for truly invalid types; empty descriptions are accepted and flagged low-confidence.
- LLM output is parsed into `TriageResponse` with `model_validate_json` + one repair retry (re-prompt with "fix JSON").

---

## 4. Step-by-Step Implementation Roadmap

### Phase 1: Environment & Data Pipeline

- [ ] Init Git + `python -m venv .venv`, create `.env.example` from §3.1, `.gitignore` (`.venv/`, `data/chroma_db/`, `.env`).
- [ ] `requirements.txt` as §3.2. Verify `PersistentClient` import.
- [ ] **Fix existing `scripts/fetch_godot_issues.py`:** Already handles `GITHUB_TOKEN`, pagination, `pull_request` skip, `topic:` → `subsystem`, `duplicate` label, rate-limit `403` sleep. Add: `per_page=100`, `time.sleep(0.5)` politeness, `len(cleaned_desc) < 50` filter — all already present. Action: add `argparse` for `target_count`/`output_path` and log `X-RateLimit-Remaining`.
- [ ] `src/schemas.py` as §3.3.
- [ ] `src/ingest.py`:
  - Text = `f"{title}\n\n{description}"` → truncate to **512 tokens** (not chars) via `tokenizer.encode` or `max_chars=2000` as fallback.
  - Query vs document prefix: documents embedded **without** prefix, queries **with** `BGE_QUERY_PREFIX` in `retriever.py` (BGE spec).
  - `normalize_embeddings=True`, batch_size=32.
  - **Upsert semantics:** `collection.upsert(ids=[bug_id], documents=[text], metadatas=[{subsystem, bug_id, state}], embeddings=[...])` — re-ingest is idempotent, not duplicate.
  - Metadata fields indexed: `subsystem`, `bug_id`, `state`.

### Phase 2: Vector Retrieval & Reasoning Service

- [ ] `src/retriever.py`:
  - `PersistentClient(path=CHROMA_PATH).get_or_create_collection("qc_bugs", metadata={"hnsw:space": "cosine"})`
  - `query_embeddings = model.encode([BGE_QUERY_PREFIX + query_text], normalize_embeddings=True)`
  - `collection.query(query_embeddings=..., n_results=TOP_K, where={"subsystem": subsystem} if subsystem != "All" else None)`
  - Pre-warm model on startup (one dummy encode) to avoid cold-start >250ms.
  - Return `list[RetrievedBug]` with `distance` (cosine).
- [ ] `src/triage.py`:
  - System prompt: "You are a Game Engine QC Specialist. Given the new bug and top-k historical bugs, decide duplicate, hypothesize root cause, suggest workaround. Cite bug_ids."
  - Build context block: `## Historical Issues\n{k}. {bug_id} [{subsystem}] distance={d:.3f}: {title} — {description[:400]}`
  - LiteLLM call: `completion(model=LLM_MODEL, messages=[...], response_format={"type": "json_object"}, timeout=LLM_TIMEOUT)` with `num_retries`.
  - **Threshold gate (before LLM):** if `min_distance > DISTANCE_THRESHOLD` → return `is_duplicate=False, confidence=0.3, duplicate_of=None` without LLM call. Log gate hit.
  - **JSON repair:** if `ValidationError`, one retry with "Your previous output was not valid JSON against this schema: ... Fix it."
- [ ] **Early eval loop (moved from Phase 4):** Run `eval/evaluate_retrieval.py` on 10-pair smoke set to tune `DISTANCE_THRESHOLD` (sweep 0.25–0.45). Do not wait for full 30-pair set.

### Phase 3: API Layer & Containerization

- [ ] `src/main.py`:
  - `GET /health` → `{"status": "ok", "chroma": "connected"|"disconnected", "model": EMBEDDING_MODEL, "llm": LLM_MODEL}`
  - `POST /api/v1/triage` → `BugReport` in, runs `retriever` → `triage` pipeline, returns `TriageResponse`. No auth required.
  - `POST /api/v1/ingest` → `IngestPayload` in, requires `X-API-Key == API_KEY` if `API_KEY` is set, runs `ingest.upsert`. Returns `{"ingested": n, "collection_count": m}`.
  - Middleware: `structlog` request id + latency logging, CORS.
  - Exception handlers: `ChromaError → 503`, `LiteLLM Timeout → 504 with fallback non-LLM response`, `ValidationError → 422`, catch-all → `500` with `request_id` (never leak stack).
- [ ] `Dockerfile` (single-stage):
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY src/ src/
  COPY data/ data/
  CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- [ ] `docker-compose.yml`:
  ```yaml
  services:
    api:
      build: .
      ports: ["8000:8000"]
      env_file: .env
      volumes: ["chroma_data:/app/data/chroma_db"]
      depends_on: [ollama]
    ollama:
      image: ollama/ollama:latest
      ports: ["11434:11434"]
      volumes: ["ollama_data:/root/.ollama"]
  volumes: { chroma_data:, ollama_data: }
  ```
  - Document `docker compose up --build` and `ollama pull llama3` (or init container).

### Phase 4: Benchmarking & Evaluation

- [ ] `eval/benchmark_dataset.json` — **Schema:**
  ```json
  [
    {
      "query_id": "GODOT-1234",
      "query_text": "Crash on shader recompilation when...",
      "query_subsystem": "Rendering",
      "relevant_ids": ["GODOT-5678", "GODOT-9012"],
      "is_duplicate": true
    }
  ]
  ```
  - 30 pairs total, manually curated from `data/raw/godot_qc_bugs.json` where `is_labeled_duplicate` or title similarity + human verification. At least 5 per subsystem.
- [ ] `eval/evaluate_retrieval.py`:
  - For each query, call `retriever.query(query_text, subsystem=query_subsystem)` and also `subsystem=None` (ablation).
  - Metrics: **HitRate@1, @3, @5**, **MRR**, **p50/p95 latency ms** (via `time.perf_counter` around encode+query, 3 warm-up runs excluded).
  - Log sweep for `DISTANCE_THRESHOLD` vs precision/recall.
  - Output: `eval/results.json` + console table. Fail CI if `HitRate@3 < 0.75`.

### Phase 5: Documentation & Presentation

- [ ] `README.md`:
  - Architecture diagram (mermaid)
  - Benchmark table (HitRate, MRR, latency) + threshold tuning chart
  - Quickstart: `cp .env.example .env && docker compose up --build` + `curl` examples for `/triage` and `/ingest`
  - API docs link (`/docs`)
  - Limitations & v2 (hybrid search, reranker `bge-reranker-base`, Qdrant alternative)

---

## 5. Acceptance Criteria

1. **Performance:** p95 vector retrieval <250ms for 1,000 docs (measured via `evaluate_retrieval.py`, warm model, CPU). Pre-warming + `onnxruntime` if needed.
2. **Quality:** HitRate@3 ≥75% and MRR ≥0.65 on 30-pair benchmark (subsystem-filtered).
3. **Reproducibility:** `docker compose up --build` builds and serves on `localhost:8000` with no manual `chroma_db` creation. `GET /health` passes.
4. **Resilience:** Malformed `description` (null, huge stack trace >10k chars, missing `subsystem`) → normalized, returns `200` triage with low confidence; never `500`. LLM timeout → `504` or graceful fallback. Ingest without `X-API-Key` when `API_KEY` set → `401`.

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| BGE cold start >250ms | Pre-warm encode on startup; batch encode; optional ONNX |
| Chroma 0.4→0.5 breaking API | Pinned `0.5.5`, `PersistentClient` abstraction in `retriever.py` |
| Ollama RAM heavy in Docker | Document `LLM_PROVIDER=groq` fallback + `OLLAMA_HOST` override |
| GitHub rate limit | `GITHUB_TOKEN` required for 500 issues; script sleeps on 403 |
| LLM JSON hallucination | Pydantic validation + one repair retry; threshold gate avoids call when irrelevant |
