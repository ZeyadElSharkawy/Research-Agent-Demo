import sys
from pathlib import Path
import pytest

# Ensure `src` is on sys.path for test discovery
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import retrieval_utils


def test_retrieve_returns_documents():
    # Simple smoke test: ensure retrieval returns at least one document for a query
    docs = retrieval_utils.retrieve_docs("Total EKYC", top_k=3)
    assert isinstance(docs, list)
    assert len(docs) >= 1, "Expected at least one document to be retrieved"
