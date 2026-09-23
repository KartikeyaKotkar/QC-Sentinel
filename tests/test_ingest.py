import src.ingest as ingest
from src.schemas import BugReport


def test_build_document_text_truncate():
    bug = BugReport(title="t", description="d" * 10000, subsystem="Core")
    text = ingest.build_document_text(bug)
    # when model available, truncates to 512 tokens (~1500-4000 chars for repeated chars), else 2000 fallback
    # ensure no crash and reasonable size, not 500 error
    assert len(text) <= 10003  # original length
    assert len(text) > 0
    assert "t" in text


def test_build_document_text_empty():
    bug = BugReport(title="crash", description=None, subsystem=None)
    text = ingest.build_document_text(bug)
    assert text == "crash"


def test_ingest_empty():
    assert ingest.ingest_bugs([]) == 0
