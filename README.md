# QC-Sentinel — RAG-Driven QA Bug Triage Engine

> **An automated Quality Control triage system for game engines.** Ingests crash logs and bug reports, finds semantic duplicates via dense retrieval, and generates root-cause hypotheses with local LLM reasoning — designed to cut triage time for QA teams.

Godot is the **reference dataset** (500+ real closed bugs from `godotengine/godot`), but the pipeline is engine-agnostic: point it at Unreal, Unity, or your proprietary engine's issue tracker and it works the same.

---

## Why this project exists

Game studios drown in duplicate bug reports. A rendering crash filed as *"LightmapGI leak inside double-sided geometry"* and *"bake incorrect indirect light"* are the same root cause, but keyword search misses them. Manual triage is slow, and cloud LLM APIs are a non-starter for unreleased titles.

QC-Sentinel solves this with a **local-first RAG pipeline**:

```
[Crash Report / Bug Text] → FastAPI → BGE Embeddings (cosine) → ChromaDB Top-5 → Distance Gate → Llama 3 (Ollama) → Structured Triage JSON
```

What it demonstrates for hiring managers: **dense retrieval + LLM orchestration + production FastAPI + Docker + evaluation** — not a chatbot wrapper.

---

## What it does

| Capability | How |
|---|---|
| **Duplicate detection** | `BAAI/bge-base-en-v1.5` (768-dim, normalized) + ChromaDB HNSW, cosine distance, Top-5 with `subsystem` metadata filtering |
| **Cost-aware gating** | If `min_distance > 0.35`, skip LLM call — return `is_duplicate: false` deterministically. Threshold tuned via sweep (0.25–0.45) |
| **Root-cause reasoning** | LiteLLM → `ollama/llama3` locally (fallback `groq/llama-3.1-8b` for CI). System prompt: *Game Engine QC Specialist* + Pydantic JSON repair |
| **Resilience** | Malformed stack traces, missing subsystem, null descriptions → normalized, never 500. LLM timeout → graceful fallback |
| **Engine-agnostic** | Any bug with `title + description + subsystem` — Godot, Unreal, Unity, custom. Swap the scraper (`scripts/fetch_godot_issues.py`) for Jira/GitHub/GitLab |

Output per report:

```json
{
  "is_duplicate": true,
  "duplicate_of": "GODOT-123745",
  "confidence_score": 0.92,
  "root_cause_hypothesis": "LightmapGI double-sided flag not respected...",
  "suggested_workaround": "Disable double-sided or use two-sided material...",
  "retrieved_context": [{ "bug_id": "GODOT-123745", "distance": 0.177, "subsystem": "rendering" }]
}
```

---

## Benchmark

| Metric | Score | Target |
|---|---|---|
| HitRate@1 | **1.00** | — |
| HitRate@3 | **1.00** | ≥0.75 |
| HitRate@5 | **1.00** | — |
| MRR | **1.00** | ≥0.65 |
| Latency p50 | **63 ms** | — |
| Latency p95 | **73 ms** | <250 ms |

*Index: 423 Godot bugs, ChromaDB, BGE 512 tokens, CPU, warm model. `eval/benchmark_dataset.json` (30 queries) is currently substring-derived — optimistic. Expect ~0.75–0.85 on paraphrased human-labeled duplicates. See `eval/results.json`.*

Honest about limitations — that's the point. The eval harness (`eval/evaluate_retrieval.py`) logs `HitRate@k`, `MRR`, `p50/p95` and runs a threshold sweep, so you can see where it breaks.

---

## Architecture

```
                ┌─────────────────────┐
                │  Any Engine's Bugs  │
                │  (title+desc+subsys)│
                └──────────┬──────────┘
                           ▼
              ┌────────────────────────┐
              │  FastAPI /api/v1/triage│  Pydantic v2 validation
              └──────────┬─────────────┘
                         ▼
              ┌────────────────────────┐
              │  Query Preprocessing   │  BGE prefix: "Represent this sentence..."
              │  512-token truncate    │  subsystem where-filter (case-insensitive)
              └──────────┬─────────────┘
                         ▼
              ┌────────────────────────┐
              │  ChromaDB (HNSW/cosine)│  BGE-base-en-v1.5, Top-5
              │  distance gate >0.35   │  → skip LLM if no close match
              └──────────┬─────────────┘
                         ▼
              ┌────────────────────────┐
              │  LiteLLM → Llama 3     │  JSON mode + Pydantic repair retry
              └──────────┬─────────────┘
                         ▼
              ┌────────────────────────┐
              │  TriageResponse JSON   │
              └────────────────────────┘
```

**Design choices worth discussing in interviews:**

- **BGE, not OpenAI embeddings** — local, no data exfiltration, no API cost. Query vs document prefix asymmetry handled correctly.
- **Dense-only v1** — proves retrieval quality before adding BM25 hybrid or `bge-reranker-base` cross-encoder (v2).
- **LiteLLM abstraction** — `LLM_PROVIDER=ollama` by default, `groq` opt-in for low-RAM CI — interview story: *"local-first but deployable"*.
- **Upsert, not insert** — `collection.upsert(ids=[bug_id])` makes re-ingest idempotent.

---

## Quickstart

### Docker (recommended)

```bash
cp .env.example .env
# optional: set GITHUB_TOKEN (for scraping), API_KEY (for /ingest auth)
docker compose up --build
# pull LLM inside ollama container (one-time, ~4GB)
docker exec -it $(docker ps -q -f name=ollama) ollama pull llama3
curl http://localhost:8000/health
# {"status":"ok","chroma":"connected","model":"BAAI/bge-base-en-v1.5","llm":"ollama/llama3","collection_count":423}
```

### Local (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch data (any engine — Godot is the example)
python scripts/fetch_godot_issues.py --count 500  # needs GITHUB_TOKEN for >60/hr

# 2. Embed + upsert
python -m src.ingest --input data/raw/godot_qc_bugs.json

# 3. Run API
uvicorn src.main:app --reload
# open http://localhost:8000/docs
```

---

## API

### `GET /health`
```bash
curl http://localhost:8000/health
```

### `POST /api/v1/triage` — no auth, for QA reporters
```bash
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Crash on shader variant warmup with MSAA 4x",
    "description": "Vulkan Forward+ backend, 4.3 stable, stack: vkCreateGraphicsPipelines failed...",
    "subsystem": "Rendering"
  }'
```

Works for any engine — just change `subsystem` to your taxonomy (`Physics`, `Animation`, `Platform_Xbox`, etc.):

```bash
# Unreal example
curl -X POST http://localhost:8000/api/v1/triage -d '{
  "title": "Nanite mesh streaming pool exhausted on large open world",
  "description": "RHI crash, streaming pool 2048MB, UObject limit...",
  "subsystem": "Rendering"
}'

# Unity example
curl -X POST http://localhost:8000/api/v1/triage -d '{
  "title": "IL2CPP build fails with Burst 1.8 on ARM64",
  "description": "Bee backend error, ARM64, Burst 1.8.3...",
  "subsystem": "Buildsystem"
}'
```

### `POST /api/v1/ingest` — requires `X-API-Key` if `API_KEY` is set
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: secret" -H "Content-Type: application/json" \
  -d '{"bugs": [{"bug_id":"MYENGINE-123","title":"new crash","description":"trace","subsystem":"Core"}]}'
```

Interactive docs: `http://localhost:8000/docs`

---

## Adapting to another engine

1. **Scraper** — fork `scripts/fetch_godot_issues.py` for Jira (`/rest/api/2/search`), GitHub (other repo), or Unity's Issue Tracker API. Keep output shape: `{bug_id, title, description, subsystem, labels}`.
2. **Subsystem taxonomy** — no code change. `subsystem` is free-form, lowercased on ingest/query for case-insensitive filter. Use `All` to disable filter.
3. **Re-ingest** — `POST /api/v1/ingest` is idempotent. Re-running with same `bug_id` updates the record.

---

## Project structure

```
├── data/raw/godot_qc_bugs.json   # 423 bugs (example dataset)
├── data/chroma_db/               # Persistent HNSW (Docker volume)
├── scripts/fetch_godot_issues.py # Pagination, rate-limit, PR filter, argparse
├── src/
│   ├── config.py                 # 12 env vars via pydantic-settings
│   ├── schemas.py                # BugReport/TriageResponse validators
│   ├── ingest.py                 # BGE 512-token truncate + upsert batch 100
│   ├── retriever.py              # BGE prefix query + where-filter
│   ├── triage.py                 # Gate + LiteLLM + JSON repair
│   └── main.py                   # FastAPI: /health, /triage, /ingest, middleware
├── tests/                        # 12 tests (ingest/retriever/api)
├── eval/
│   ├── benchmark_dataset.json    # 30 queries
│   └── evaluate_retrieval.py     # HitRate, MRR, p50/p95, threshold sweep
├── Dockerfile                    # python:3.11-slim, no venv in context
├── docker-compose.yml            # api + ollama + volumes
└── plan.md                       # Full spec + roadmap + acceptance criteria
```

---

## Configuration

See `.env.example`. Key knobs:

| Var | Default | Notes |
|---|---|---|
| `DISTANCE_THRESHOLD` | `0.35` | Tune via `eval/evaluate_retrieval.py` sweep 0.25–0.45 |
| `TOP_K` | `5` | Retrieval depth |
| `OLLAMA_HOST` | `http://ollama:11434` | Override for host Ollama |
| `LLM_PROVIDER` | `ollama` | `groq` for low-RAM CI |
| `API_KEY` | *(empty)* | If set, `/ingest` requires `X-API-Key` |

---

## Evaluation & testing

```bash
# Retrieval quality + latency
PYTHONPATH=. python eval/evaluate_retrieval.py --output eval/results.json
PYTHONPATH=. python eval/evaluate_retrieval.py --no-filter  # ablation

# Unit/API tests (no Ollama needed, LLM mocked)
PYTHONPATH=. pytest -q  # 12 passed

# Full triage with real retrieval (needs Ollama)
curl -X POST http://localhost:8000/api/v1/triage -d '{"title":"...","description":"...","subsystem":"..."}'
```

---

## Limitations & roadmap

**Honest limitations (talk about these in interviews):**

- Benchmark is substring-derived — real paraphrased duplicates will score lower. Next step: human-label 30 pairs across Unreal/Unity/Godot for a cross-engine eval set.
- Dense-only — misses exact `vkCreateGraphicsPipelines` stack trace tokens. v2 adds BM25 hybrid + `bge-reranker-base` cross-encoder.
- Scale — Chroma is fine for <10k bugs. For 100k+, move to Qdrant + sharding.
- No rate limiting on `/triage` — fine for internal QA, needs gateway for public.

**v2 ideas:** hybrid search, Qdrant, Jira webhook, Slack bot for QA, fine-tuned BGE on engine-specific corpus.

---

## Why hire me for this

This isn't a tutorial RAG. It's a **production-shaped** system:

- Pydantic v2 everywhere (validation, settings, LLM output), not `dict` soup
- Docker that actually builds (`1065s` cold, `14s` cached) with `.dockerignore` and named volumes
- Evaluation before shipping — threshold tuned, not hardcoded
- Resilience over happy path — malformed traces → 200 fallback, never 500
- Engine-agnostic from day one — Godot is data, not the product

If you're a game studio, replace `scripts/fetch_godot_issues.py` with your tracker and it triages your bugs tomorrow.
