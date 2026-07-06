import pytest
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="session")
def embedder():
    """Load the embedding model once per test session (it's slow to load)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def sample_pages():
    """Two fake pages that mimic the output of extract_text_from_pdf."""
    return [
        {"page": 1, "text": " ".join([f"word{i}" for i in range(600)])},
        {"page": 2, "text": " ".join([f"page2word{i}" for i in range(300)])},
    ]
