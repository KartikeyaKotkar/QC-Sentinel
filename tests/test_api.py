from unittest.mock import patch

from fastapi.testclient import TestClient

import src.ingest
from src.main import app
from src.schemas import TriageResponse

client = TestClient(app)


def test_health():
    with patch.object(src.ingest, "get_collection") as mc:
        mc.return_value.count.return_value = 5
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["chroma"] == "connected"


def test_health_degraded():
    with patch.object(src.ingest, "get_collection", side_effect=RuntimeError("down")):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"


def test_triage_malformed_normalized():
    # null description/subsystem -> 200 not 500
    fake = TriageResponse(is_duplicate=False, duplicate_of=None, confidence_score=0.3, root_cause_hypothesis="x", suggested_workaround="y", retrieved_context=[])
    async def fake_atriage(bug, retrieved=None):
        return fake
    with patch("src.triage.atriage_bug", fake_atriage):
        r = client.post("/api/v1/triage", json={"title": "crash", "description": None, "subsystem": None})
        assert r.status_code == 200


def test_triage_empty_title_422():
    r = client.post("/api/v1/triage", json={"title": "", "description": "d"})
    assert r.status_code == 422


def test_ingest_auth():
    from src import config
    orig = config.settings.API_KEY
    config.settings.API_KEY = "secret"
    try:
        r = client.post("/api/v1/ingest", json={"bugs": [{"title": "t", "description": "d"}]})
        assert r.status_code == 401
        with patch.object(src.ingest, "ingest_bugs", return_value=1), patch.object(src.ingest, "get_collection") as mc:
            mc.return_value.count.return_value = 1
            r = client.post("/api/v1/ingest", headers={"X-API-Key": "secret"}, json={"bugs": [{"title": "t", "description": "d"}]})
            assert r.status_code == 200
            assert r.json()["ingested"] == 1
    finally:
        config.settings.API_KEY = orig
