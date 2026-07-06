from papers.pipeline import build_context, ask_gemini


def test_build_context_joins_chunks_with_separator():
    chunks = ["First chunk of text.", "Second chunk of text.", "Third chunk."]
    result = build_context(chunks)
    assert "First chunk" in result
    assert "Third chunk" in result
    assert result.count("---") == 2


def test_build_context_empty_list():
    assert build_context([]) == ""


def test_ask_gemini_returns_demo_when_no_api_key(settings):
    """Without an API key, ask_gemini must fall back to demo mode."""
    settings.GEMINI_API_KEY = ""
    result = ask_gemini("What is this paper about?", "Some context", mode="qa")
    assert "DEMO MODE" in result
