import faiss
import numpy as np

from papers.pipeline import EMBEDDING_DIM


def test_faiss_index_returns_nearest_neighbour(embedder):
    """Build a tiny in-memory index and confirm the closest chunk is retrieved."""
    corpus = [
        "Transformers use self-attention to process sequences.",
        "Bananas grow on tropical trees in humid climates.",
        "Convolutional neural networks are used for image recognition.",
    ]
    embeddings = embedder.encode(corpus, normalize_embeddings=True).astype("float32")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    query = embedder.encode(
        ["What is self-attention in language models?"],
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(query, k=1)
    assert ids[0][0] == 0


def test_faiss_scores_ordered_descending(embedder):
    """Inner-product FAISS returns scores in descending order."""
    corpus = ["cat", "dog", "car", "banana"]
    embeddings = embedder.encode(corpus, normalize_embeddings=True).astype("float32")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    query = embedder.encode(["pet animal"], normalize_embeddings=True).astype("float32")
    scores, _ = index.search(query, k=4)
    assert list(scores[0]) == sorted(scores[0], reverse=True)
