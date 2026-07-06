from papers.pipeline import chunk_pages, CHUNK_SIZE, CHUNK_OVERLAP


def test_chunk_pages_returns_list_of_dicts(sample_pages):
    chunks = chunk_pages(sample_pages)
    assert isinstance(chunks, list)
    assert all(isinstance(c, dict) for c in chunks)
    assert all({"text", "page", "chunk_index"} <= c.keys() for c in chunks)


def test_chunk_size_respected(sample_pages):
    """No chunk should exceed CHUNK_SIZE words."""
    chunks = chunk_pages(sample_pages)
    for c in chunks:
        assert len(c["text"].split()) <= CHUNK_SIZE


def test_chunk_overlap_is_correct(sample_pages):
    """Consecutive chunks from the same page should overlap by CHUNK_OVERLAP words."""
    chunks = chunk_pages(sample_pages)
    page1_chunks = [c for c in chunks if c["page"] == 1]
    assert len(page1_chunks) >= 2
    tail = page1_chunks[0]["text"].split()[-CHUNK_OVERLAP:]
    head = page1_chunks[1]["text"].split()[:CHUNK_OVERLAP]
    assert tail == head


def test_chunk_indices_are_sequential(sample_pages):
    """chunk_index should count up from 0 with no gaps."""
    chunks = chunk_pages(sample_pages)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_empty_pages_returns_empty(sample_pages):
    assert chunk_pages([]) == []
