from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.schemas import BugReport, HealthResponse, IngestPayload, IngestResponse, TriageResponse

try:
    import structlog

    _log = structlog.get_logger(__name__)
except ImportError:
    import logging

    _log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from src.ingest import get_model

        await asyncio.to_thread(get_model)
        _log.info("model warmup complete")
    except Exception as exc:
        _log.warning("model warmup failed: %s", exc)
    yield


app = FastAPI(title="QC-Sentinel", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = rid
    try:
        import structlog.contextvars

        structlog.contextvars.bind_contextvars(request_id=rid)
    except Exception:
        pass
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        try:
            _log.error(
                "request failed",
                request_id=rid,
                path=request.url.path,
                method=request.method,
                latency_ms=round(latency, 2),
                error=str(exc),
            )
        except Exception:
            pass
        raise
    latency = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    try:
        _log.info(
            "request",
            request_id=rid,
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=round(latency, 2),
        )
    except Exception:
        pass
    try:
        import structlog.contextvars

        structlog.contextvars.unbind_contextvars("request_id")
    except Exception:
        pass
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    rid = getattr(request.state, "request_id", request.headers.get("X-Request-ID") or uuid.uuid4().hex)
    headers = dict(exc.headers) if exc.headers else {}
    headers["X-Request-ID"] = rid
    if exc.status_code in (500, 503):
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.detail, "request_id": rid}, headers=headers
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=headers)


@app.exception_handler(Exception)
async def catch_all_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", request.headers.get("X-Request-ID") or uuid.uuid4().hex)
    mod = type(exc).__module__.lower()
    name = type(exc).__name__.lower()
    is_chroma = "chroma" in mod or "chroma" in name or "chromadb" in mod
    if is_chroma:
        try:
            _log.error("chroma error", request_id=rid, error=str(exc))
        except Exception:
            pass
        return JSONResponse(
            status_code=503,
            content={"detail": "Vector store unavailable", "request_id": rid},
            headers={"X-Request-ID": rid},
        )
    try:
        _log.error("unhandled error", request_id=rid, error=str(exc))
    except Exception:
        pass
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": rid},
        headers={"X-Request-ID": rid},
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    try:
        from src.ingest import get_collection

        collection = get_collection()
        count = collection.count()
        return HealthResponse(
            status="ok",
            chroma="connected",
            model=settings.EMBEDDING_MODEL,
            llm=settings.LLM_MODEL,
            collection_count=count,
        )
    except Exception as exc:
        try:
            _log.warning("health check chroma disconnected: %s", exc)
        except Exception:
            pass
        return HealthResponse(
            status="degraded",
            chroma="disconnected",
            model=settings.EMBEDDING_MODEL,
            llm=settings.LLM_MODEL,
            collection_count=None,
        )


@app.post("/api/v1/triage", response_model=TriageResponse)
async def triage_endpoint(bug: BugReport, request: Request):
    try:
        from src.triage import atriage_bug

        result = await atriage_bug(bug)
        return result
    except Exception as exc:
        mod = type(exc).__module__.lower()
        name = type(exc).__name__.lower()
        is_chroma = "chroma" in mod or "chroma" in name or "chromadb" in mod
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        if is_chroma:
            raise HTTPException(status_code=503, detail="Vector store unavailable")
        _log.error("triage failed", request_id=rid, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    payload: IngestPayload,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if settings.API_KEY:
        if x_api_key != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        from src.ingest import get_collection, ingest_bugs

        n = await asyncio.to_thread(ingest_bugs, payload.bugs)
        try:
            m = await asyncio.to_thread(lambda: get_collection().count())
        except Exception:
            m = n
        return IngestResponse(ingested=n, collection_count=m)
    except HTTPException:
        raise
    except Exception as exc:
        mod = type(exc).__module__.lower()
        name = type(exc).__name__.lower()
        is_chroma = "chroma" in mod or "chroma" in name or "chromadb" in mod
        if is_chroma:
            raise HTTPException(status_code=503, detail="Vector store unavailable")
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        _log.error("ingest failed", request_id=rid, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal server error")
