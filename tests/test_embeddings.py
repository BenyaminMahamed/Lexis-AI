import numpy as np
import pytest

from papers.pipeline import EMBEDDING_DIM


def test_embedding_dimension(embedder):
    vec = embedder.encode(["hello world"], normalize_embeddings=True)
    assert vec.shape == (1, EMBEDDING_DIM)


def test_embedding_is_normalised(embedder):
    """L2 norm of a normalised embedding should be ~1.0."""
    vec = embedder.encode(["some academic text"], normalize_embeddings=True)
    norm = np.linalg.norm(vec[0])
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_similar_texts_have_higher_similarity_than_unrelated(embedder):
    """Semantic sanity check: related sentences score closer than unrelated ones."""
    vecs = embedder.encode(
        [
            "The paper introduces a new transformer architecture",
            "This work proposes a novel transformer model",
            "Bananas are yellow tropical fruits",
        ],
        normalize_embeddings=True,
    )
    similar = float(np.dot(vecs[0], vecs[1]))
    unrelated = float(np.dot(vecs[0], vecs[2]))
    assert similar > unrelated
