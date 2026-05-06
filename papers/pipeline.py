"""
RAG Pipeline — Gemini-powered
PDF → text extraction → chunking → embeddings → FAISS
User query → embed → FAISS search → retrieve chunks → Gemini → answer
"""

import json
import logging
import numpy as np
from pathlib import Path

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
from google.genai import types

from django.conf import settings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
GEMINI_MODEL = "gemini-2.0-flash"

INDEX_PATH = Path(settings.FAISS_INDEX_PATH)
META_PATH = INDEX_PATH / 'metadata.json'
INDEX_FILE = INDEX_PATH / 'index.bin'


# ── Embedding model ────────────────────────────────────────────────────────────

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformers model...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


# ── PDF Extraction ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> list:
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({'page': page_num + 1, 'text': text})
    doc.close()
    return pages


def extract_title(file_path: str) -> str:
    doc = fitz.open(file_path)
    first_page = doc[0].get_text().strip()
    doc.close()
    lines = [l.strip() for l in first_page.split('\n') if l.strip()]
    return lines[0][:200] if lines else "Untitled"


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_pages(pages: list) -> list:
    chunks = []
    idx = 0
    for page_data in pages:
        words = page_data['text'].split()
        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunks.append({
                'text': ' '.join(words[start:end]),
                'page': page_data['page'],
                'chunk_index': idx,
            })
            idx += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── FAISS Index ────────────────────────────────────────────────────────────────

def _load_or_create_index():
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    if INDEX_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
        with open(META_PATH) as f:
            metadata = json.load(f)
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        metadata = []
    return index, metadata


def _save_index(index, metadata):
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_PATH, 'w') as f:
        json.dump(metadata, f)


def add_chunks_to_index(chunks_with_ids: list) -> list:
    model = get_embedding_model()
    texts = [c['text'] for c in chunks_with_ids]
    embeddings = model.encode(texts, normalize_embeddings=True).astype('float32')

    index, metadata = _load_or_create_index()
    start_id = index.ntotal
    index.add(embeddings)

    faiss_ids = list(range(start_id, start_id + len(chunks_with_ids)))
    for i, c in enumerate(chunks_with_ids):
        metadata.append({
            'faiss_id': faiss_ids[i],
            'chunk_db_id': c['chunk_db_id'],
            'paper_id': c['paper_id'],
        })

    _save_index(index, metadata)
    return faiss_ids


def search_index(query: str, paper_ids=None, top_k: int = TOP_K):
    if not INDEX_FILE.exists():
        return []

    model = get_embedding_model()
    query_vec = model.encode([query], normalize_embeddings=True).astype('float32')

    index, metadata = _load_or_create_index()
    if index.ntotal == 0:
        return []

    k = min(index.ntotal, top_k * 10 if paper_ids else top_k)
    scores, faiss_ids = index.search(query_vec, k)

    results = []
    for score, fid in zip(scores[0], faiss_ids[0]):
        if fid == -1:
            continue
        meta = next((m for m in metadata if m['faiss_id'] == fid), None)
        if meta is None:
            continue
        if paper_ids and meta['paper_id'] not in paper_ids:
            continue
        results.append({
            'chunk_db_id': meta['chunk_db_id'],
            'paper_id': meta['paper_id'],
            'score': float(score),
        })
        if len(results) >= top_k:
            break

    return results


# ── Gemini LLM Layer ───────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    'qa': (
        "You are a precise academic research assistant. Answer the user's question "
        "using ONLY the provided context from the paper. If the answer isn't clearly "
        "supported by the context, say so. Be concise, accurate, and direct."
    ),
    'summarise': (
        "You are an expert academic summariser. Based on the provided context, produce "
        "a structured summary with these sections:\n"
        "**Main Contribution** — what the paper introduces or proves\n"
        "**Methodology** — how they did it\n"
        "**Key Findings** — the main results\n"
        "**Limitations** — weaknesses acknowledged or apparent\n"
        "Be specific to this paper, not generic."
    ),
    'critique': (
        "You are a rigorous peer reviewer for a top-tier academic conference. "
        "Based on the provided context, produce a structured critique:\n"
        "**Core Assumptions** — what the authors take for granted\n"
        "**Methodological Weaknesses** — flaws in experimental design or evaluation\n"
        "**Missing Baselines / Experiments** — what comparisons are absent\n"
        "**Suggested Improvements** — concrete ways to strengthen the work\n"
        "Be specific and critical. Avoid vague praise."
    ),
    'compare': (
        "You are a comparative research analyst. Based on the provided context from "
        "multiple papers, produce a structured comparison covering: approach differences, "
        "methodology, strengths/weaknesses per paper, and a final verdict on which is stronger."
    ),
}


def build_context(chunk_texts: list) -> str:
    return "\n\n---\n\n".join(chunk_texts)


def ask_gemini(question: str, context: str, mode: str = 'qa') -> str:
    api_key = getattr(settings, 'GEMINI_API_KEY', '')
    if not api_key:
        return _demo_response(mode, context)

    client = genai.Client(api_key=api_key)
    system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS['qa'])

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Context from paper(s):\n\n{context}\n\n---\n\nQuestion: {question}",
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )
        return response.text
    except Exception as e:
        error_str = str(e)
        if '429' in error_str or 'quota' in error_str.lower():
            return "⚠️ API quota exceeded. Please wait a moment and try again, or check your Gemini API key limits at aistudio.google.com."
        elif '404' in error_str or 'not found' in error_str.lower():
            return "⚠️ Model not available. Please check your Gemini API configuration."
        elif '401' in error_str or 'api key' in error_str.lower():
            return "⚠️ Invalid API key. Please check your GEMINI_API_KEY in .env."
        else:
            logger.error(f"Gemini API error: {e}")
            return "⚠️ An error occurred while generating a response. Please try again."


# Backward-compatible alias
ask_llm = ask_gemini


def _demo_response(mode: str, context: str) -> str:
    preview = context[:500].replace('\n', ' ')
    word_count = len(context.split())
    return (
        f"**DEMO MODE** — No `GEMINI_API_KEY` configured.\n\n"
        f"The RAG pipeline retrieved {word_count} words of context successfully:\n\n"
        f"_{preview}..._\n\n"
        f"Add `GEMINI_API_KEY=your_key` to `.env` to get real responses."
    )


# ── Full Ingestion Pipeline ────────────────────────────────────────────────────

def process_paper(paper_db_obj) -> int:
    from papers.models import Chunk

    file_path = paper_db_obj.uploaded_file.path

    if not paper_db_obj.title:
        paper_db_obj.title = extract_title(file_path)
        paper_db_obj.save()

    pages = extract_text_from_pdf(file_path)
    if not pages:
        raise ValueError("Could not extract text from PDF.")

    chunks = chunk_pages(pages)

    db_chunks = []
    for c in chunks:
        db_chunk = Chunk.objects.create(
            paper=paper_db_obj,
            text=c['text'],
            chunk_index=c['chunk_index'],
            page_number=c['page'],
        )
        db_chunks.append(db_chunk)

    chunks_with_ids = [
        {'text': db.text, 'chunk_db_id': db.id, 'paper_id': paper_db_obj.id}
        for db in db_chunks
    ]
    faiss_ids = add_chunks_to_index(chunks_with_ids)

    for db_chunk, fid in zip(db_chunks, faiss_ids):
        db_chunk.faiss_id = fid
        db_chunk.save(update_fields=['faiss_id'])

    paper_db_obj.processed = True
    paper_db_obj.chunk_count = len(db_chunks)
    paper_db_obj.save()

    return len(db_chunks)


def answer_question(question: str, paper_ids: list, mode: str = 'qa') -> dict:
    from papers.models import Chunk

    results = search_index(question, paper_ids=paper_ids)
    if not results:
        return {'answer': 'No relevant content found in the selected papers.', 'sources': []}

    chunk_db_ids = [r['chunk_db_id'] for r in results]
    chunks = Chunk.objects.filter(id__in=chunk_db_ids).select_related('paper')
    chunk_map = {c.id: c for c in chunks}
    ordered_chunks = [chunk_map[cid] for cid in chunk_db_ids if cid in chunk_map]

    context = build_context([c.text for c in ordered_chunks])
    answer = ask_gemini(question, context, mode=mode)

    sources = [
        {
            'text': c.text[:300] + '...',
            'page': c.page_number,
            'paper_id': c.paper_id,
            'paper_title': c.paper.title,
        }
        for c in ordered_chunks
    ]

    return {'answer': answer, 'sources': sources}