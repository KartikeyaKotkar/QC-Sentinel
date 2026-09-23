from unittest.mock import MagicMock, patch

from src.retriever import query_bugs


def test_query_empty_text():
    assert query_bugs("") == []
    assert query_bugs("   ") == []


def test_query_empty_collection():
    with patch("src.retriever.get_collection") as mock_col, patch("src.retriever.get_model"):
        mock_col.return_value.count.return_value = 0
        assert query_bugs("crash shader") == []


def test_query_where_filter():
    with patch("src.retriever.get_collection") as mock_col, patch("src.retriever.get_model") as mock_model:
        mock_col.return_value.count.return_value = 10
        mock_model.return_value.encode.return_value.tolist.return_value = [[0.1] * 768]
        mock_col.return_value.query.return_value = {
            "ids": [["GODOT-1"]],
            "documents": [["title\n\ndesc"]],
            "metadatas": [[{"bug_id": "GODOT-1", "subsystem": "Rendering"}]],
            "distances": [[0.2]],
        }
        res = query_bugs("test", subsystem="Rendering")
        assert len(res) == 1
        assert res[0].subsystem == "Rendering"
        # where lowercased for case-insensitive match
        assert mock_col.return_value.query.call_args.kwargs["where"] == {"subsystem": "rendering"}

        res = query_bugs("test", subsystem="All")
        assert mock_col.return_value.query.call_args.kwargs["where"] is None


def test_query_chroma_error_returns_empty():
    with patch("src.retriever.get_collection", side_effect=RuntimeError("chroma down")):
        assert query_bugs("test") == []
