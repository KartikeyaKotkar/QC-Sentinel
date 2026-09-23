from __future__ import annotations

import json
import os

from pydantic import ValidationError

from src.config import settings
from src.retriever import query_bugs
from src.schemas import BugReport, RetrievedBug, TriageResponse

try:
    import structlog

    _log = structlog.get_logger(__name__)
except ImportError:
    import logging

    _log = logging.getLogger(__name__)

try:
    import litellm  # type: ignore
except ImportError:
    litellm = None  # type: ignore

SYSTEM_PROMPT = (
    "You are a Game Engine QC Specialist. Given the new bug and top-k historical "
    "bugs, decide duplicate, hypothesize root cause, suggest workaround. Cite bug_ids. "
    "Respond ONLY with valid JSON matching schema."
)

_FALLBACK_ROOT_CAUSE = "No close historical match found for this bug; manual triage required."
_FALLBACK_WORKAROUND = "Treat as new issue; collect reproduction steps and assign to subsystem owner."


def _build_context(retrieved: list[RetrievedBug]) -> str:
    if not retrieved:
        return "## Historical Issues\nNo historical issues found."
    lines = ["## Historical Issues"]
    for k, r in enumerate(retrieved, 1):
        desc = (r.description or "")[:400]
        lines.append(f"{k}. {r.bug_id} [{r.subsystem}] distance={r.distance:.3f}: {r.title} \u2014 {desc}")
    return "\n".join(lines)


def _build_user_prompt(bug: BugReport, context_block: str) -> str:
    return (
        f"{context_block}\n\n"
        f"## New Bug\n"
        f"bug_id: {bug.bug_id}\n"
        f"subsystem: {bug.subsystem}\n"
        f"title: {bug.title}\n"
        f"description: {bug.description or ''}\n\n"
        f"Task: Decide if the new bug is a duplicate of any historical issue. "
        f"Return JSON with keys: is_duplicate (bool), duplicate_of (string|null), "
        f"confidence_score (0.0-1.0), root_cause_hypothesis (string), suggested_workaround (string)."
    )


def _fallback(retrieved: list[RetrievedBug], reason: str = "") -> TriageResponse:
    if reason:
        _log.warning("triage fallback: %s", reason)
    return TriageResponse(
        is_duplicate=False,
        duplicate_of=None,
        confidence_score=0.3,
        root_cause_hypothesis=_FALLBACK_ROOT_CAUSE,
        suggested_workaround=_FALLBACK_WORKAROUND,
        retrieved_context=retrieved,
    )


def _extract_content(response) -> str:
    """Extract text content from litellm response (supports multiple shapes)."""
    try:
        return response.choices[0].message.content or ""  # type: ignore
    except Exception:
        pass
    try:
        return response["choices"][0]["message"]["content"]  # type: ignore
    except Exception:
        return str(response)


def _call_llm(messages: list[dict]) -> str | None:
    if litellm is None:
        _log.warning("litellm not installed — using fallback")
        return None

    # Ensure ollama host is set for litellm ollama provider
    if settings.OLLAMA_HOST and settings.LLM_MODEL.startswith("ollama/"):
        os.environ.setdefault("OLLAMA_API_BASE", settings.OLLAMA_HOST)

    last_exc: Exception | None = None
    for attempt in range(settings.LLM_MAX_RETRIES + 1):
        try:
            kwargs: dict = dict(
                model=settings.LLM_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=settings.LLM_TIMEOUT,
            )
            # Pass api_base for ollama models when host is configured
            if settings.LLM_MODEL.startswith("ollama/") and settings.OLLAMA_HOST:
                kwargs["api_base"] = settings.OLLAMA_HOST
            resp = litellm.completion(**kwargs)  # type: ignore
            content = _extract_content(resp)
            if content:
                return content
            last_exc = ValueError("empty LLM response")
        except Exception as exc:
            last_exc = exc
            # timeout / connection errors are expected — log and retry
            _log.warning("LLM call failed attempt %d/%d: %s", attempt + 1, settings.LLM_MAX_RETRIES + 1, exc)
            # retry only on transient errors; still retry up to max
            continue
    if last_exc:
        _log.warning("LLM unreachable after retries: %s", last_exc)
    return None


def _parse_output(output: str, retrieved: list[RetrievedBug]) -> TriageResponse | None:
    """Try to validate output as TriageResponse. Returns None on failure."""
    # First attempt: direct validation
    try:
        # Allow both json_object string and plain dict string
        validated = TriageResponse.model_validate_json(output)
        validated.retrieved_context = retrieved
        return validated
    except ValidationError as ve:
        _log.warning("LLM output validation failed: %s — attempting repair", ve)
    except Exception as exc:
        _log.warning("LLM output parse failed: %s", exc)

    # Repair retry: ask LLM to fix JSON
    if litellm is None:
        return None
    schema = json.dumps(TriageResponse.model_json_schema(), indent=2)
    repair_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Your previous output was not valid JSON against this schema: {schema}. Fix it. Previous: {output}",
        },
    ]
    repaired = _call_llm(repair_messages)
    if not repaired:
        return None
    try:
        validated = TriageResponse.model_validate_json(repaired)
        validated.retrieved_context = retrieved
        return validated
    except (ValidationError, Exception) as exc:
        _log.warning("LLM repair output still invalid: %s", exc)
        # Last resort: try json.loads then model_validate
        try:
            data = json.loads(repaired)
            validated = TriageResponse.model_validate(data)
            validated.retrieved_context = retrieved
            return validated
        except Exception:
            return None


def triage_bug(bug: BugReport, retrieved: list[RetrievedBug] | None = None) -> TriageResponse:
    # Retrieve if not provided
    if retrieved is None:
        try:
            query_text = f"{bug.title}\n\n{bug.description or ''}"
            retrieved = query_bugs(query_text, subsystem=bug.subsystem)
        except Exception as exc:
            _log.warning("retrieval failed: %s", exc)
            retrieved = []

    retrieved = retrieved or []

    # Threshold gate BEFORE LLM
    if not retrieved:
        _log.info("triage threshold gate: no retrieved context")
        return _fallback(retrieved, "no retrieved context")

    min_dist = min(r.distance for r in retrieved)
    if min_dist > settings.DISTANCE_THRESHOLD:
        _log.info("triage threshold gate: min_distance %.3f > %.3f", min_dist, settings.DISTANCE_THRESHOLD)
        return TriageResponse(
            is_duplicate=False,
            duplicate_of=None,
            confidence_score=0.3,
            root_cause_hypothesis=_FALLBACK_ROOT_CAUSE,
            suggested_workaround=_FALLBACK_WORKAROUND,
            retrieved_context=retrieved,
        )

    # Build prompt and call LLM
    context_block = _build_context(retrieved)
    user_prompt = _build_user_prompt(bug, context_block)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    output = _call_llm(messages)
    if output is None:
        return _fallback(retrieved, "LLM unreachable or not installed")

    result = _parse_output(output, retrieved)
    if result is not None:
        return result

    # If parsing/repair failed, graceful fallback (never 500)
    _log.warning("triage: returning fallback after parse failure")
    return _fallback(retrieved, "LLM output parse failed")


async def atriage_bug(bug: BugReport, retrieved: list[RetrievedBug] | None = None) -> TriageResponse:
    """Async wrapper — runs sync triage_bug in thread if needed, or uses acompletion when available."""
    if retrieved is None:
        try:
            query_text = f"{bug.title}\n\n{bug.description or ''}"
            retrieved = query_bugs(query_text, subsystem=bug.subsystem)
        except Exception as exc:
            _log.warning("retrieval failed: %s", exc)
            retrieved = []
    retrieved = retrieved or []

    if not retrieved:
        return _fallback(retrieved, "no retrieved context")
    min_dist = min(r.distance for r in retrieved)
    if min_dist > settings.DISTANCE_THRESHOLD:
        return TriageResponse(
            is_duplicate=False,
            duplicate_of=None,
            confidence_score=0.3,
            root_cause_hypothesis=_FALLBACK_ROOT_CAUSE,
            suggested_workaround=_FALLBACK_WORKAROUND,
            retrieved_context=retrieved,
        )

    context_block = _build_context(retrieved)
    user_prompt = _build_user_prompt(bug, context_block)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Prefer litellm.acompletion if available
    if litellm is not None and hasattr(litellm, "acompletion"):
        if settings.OLLAMA_HOST and settings.LLM_MODEL.startswith("ollama/"):
            os.environ.setdefault("OLLAMA_API_BASE", settings.OLLAMA_HOST)
        last_exc: Exception | None = None
        output: str | None = None
        for attempt in range(settings.LLM_MAX_RETRIES + 1):
            try:
                kwargs: dict = dict(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=settings.LLM_TIMEOUT,
                )
                if settings.LLM_MODEL.startswith("ollama/") and settings.OLLAMA_HOST:
                    kwargs["api_base"] = settings.OLLAMA_HOST
                resp = await litellm.acompletion(**kwargs)  # type: ignore
                output = _extract_content(resp)
                if output:
                    break
            except Exception as exc:
                last_exc = exc
                _log.warning("async LLM call failed attempt %d: %s", attempt + 1, exc)
        if output is None:
            if last_exc:
                _log.warning("async LLM unreachable: %s", last_exc)
            return _fallback(retrieved, "async LLM unreachable")
        result = _parse_output(output, retrieved)
        if result is not None:
            return result
        return _fallback(retrieved, "async parse failed")

    # Fallback to sync path in thread
    import asyncio

    return await asyncio.to_thread(triage_bug, bug, retrieved)
