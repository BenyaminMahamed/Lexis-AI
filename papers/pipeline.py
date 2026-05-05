"""
RAG Pipeline
------------
PDF → text extraction → chunking → embeddings → FAISS
User query → embed → FAISS search → retrieve chunks → LLM → answer
"""

import os
import json
import logging
import numpy as np
from pathlib import Path

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import openai

from django.conf import settings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CHUNK_SIZE = 500        # words per chunk
CHUNK_OVERLAP = 50      # word overlap between chunks
TOP_K = 4               # number of chunks to retrieve
EMBEDDING_DIM = 384     # matches 'all-MiniLM-L6-v2'
INDEX_PATH = Path(settings.FAISS_INDEX_PATH)
META_PATH = INDEX_PATH / 'metadata.json'
INDEX_FILE = INDEX_PATH / 'index.bin'

# ── Singleton model loader ─────────────────────────────────────────────────────

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformers model...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


# ── PDF Extraction ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Returns list of {page: int, text: str} dicts.
    """
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({'page': page_num + 1, 'text': text})
    doc.close()
    return pages


def extract_title(file_path: str) -> str:
    """Best-effort title extraction from first page."""
    doc = fitz.open(file_path)
    first_page = doc[0].get_text().strip()
    doc.close()
    lines = [l.strip() for l in first_page.split('\n') if l.strip()]
    return lines[0][:200] if lines else "Untitled"


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Splits page text into overlapping word-based chunks.
    Returns list of {text, page, chunk_index}.
    """
    chunks = []
    idx = 0
    for page_data in pages:
        words = page_data['text'].split()
        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunk_text = ' '.join(words[start:end])
            chunks.append({
                'text': chunk_text,
                'page': page_data['page'],
                'chunk_index': idx,
            })
            idx += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── FAISS Index Management ─────────────────────────────────────────────────────

def _load_or_create_index():
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    if INDEX_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
        with open(META_PATH) as f:
            metadata = json.load(f)
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product (cosine on normalised vecs)
        metadata = []  # list of {chunk_db_id, paper_id}
    return index, metadata


def _save_index(index, metadata):
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_PATH, 'w') as f:
        json.dump(metadata, f)


def add_chunks_to_index(chunks_with_ids: list[dict]) -> list[int]:
    """
    chunks_with_ids: list of {text, chunk_db_id, paper_id}
    Returns list of faiss_ids assigned.
    """
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


def search_index(query: str, paper_ids: list[int] = None, top_k: int = TOP_K):
    """
    Returns list of {chunk_db_id, paper_id, score} sorted by relevance.
    paper_ids: if provided, filter to only those papers.
    """
    if not INDEX_FILE.exists():
        return []

    model = get_embedding_model()
    query_vec = model.encode([query], normalize_embeddings=True).astype('float32')

    index, metadata = _load_or_create_index()
    if index.ntotal == 0:
        return []

    # Search more than top_k so we can filter by paper_id if needed
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


# ── LLM Layer ──────────────────────────────────────────────────────────────────

def build_context(chunk_texts: list[str]) -> str:
    return "\n\n---\n\n".join(chunk_texts)


def ask_llm(question: str, context: str, mode: str = 'qa') -> str:
    """
    mode: 'qa' | 'summarise' | 'critique' | 'compare'
    """
    system_prompts = {
        'qa': (
            "You are a precise academic research assistant. Answer the user's question "
            "using ONLY the provided context from the paper. If the answer isn't in the "
            "context, say so clearly. Be concise and accurate."
        ),
        'summarise': (
            "You are an academic summariser. Based on the provided context, produce a "
            "clear summary with 5-6 bullet points covering: main contribution, methodology, "
            "key findings, and limitations. Be precise, not generic."
        ),
        'critique': (
            "You are a rigorous academic peer reviewer. Based on the provided context, "
            "critique the paper by identifying: core assumptions, methodological weaknesses, "
            "missing experiments or baselines, and potential improvements. Be specific."
        ),
        'compare': (
            "You are a comparative research analyst. Based on the provided context from "
            "multiple papers, compare their approaches, methodologies, strengths, and "
            "weaknesses. Structure your response clearly."
        ),
    }

    api_key = settings.OPENAI_API_KEY
    if not api_key:
        return _fallback_response(mode, context)

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': system_prompts.get(mode, system_prompts['qa'])},
            {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    return response.choices[0].message.content


def _fallback_response(mode: str, context: str) -> str:
    """Used when no OpenAI key is set — returns raw chunks for dev testing."""
    return f"[DEV MODE — no OpenAI key set]\n\nRetrieved context:\n\n{context[:1000]}..."


# ── Full Pipeline ──────────────────────────────────────────────────────────────

def process_paper(paper_db_obj) -> int:
    """
    Full ingestion pipeline for a Paper model instance.
    Returns number of chunks created.
    """
    from papers.models import Chunk

    file_path = paper_db_obj.uploaded_file.path

    # 1. Extract title if not set
    if not paper_db_obj.title:
        paper_db_obj.title = extract_title(file_path)
        paper_db_obj.save()

    # 2. Extract text
    pages = extract_text_from_pdf(file_path)
    if not pages:
        raise ValueError("Could not extract text from PDF.")

    # 3. Chunk
    chunks = chunk_pages(pages)

    # 4. Save chunks to DB first (to get IDs)
    db_chunks = []
    for c in chunks:
        db_chunk = Chunk.objects.create(
            paper=paper_db_obj,
            text=c['text'],
            chunk_index=c['chunk_index'],
            page_number=c['page'],
        )
        db_chunks.append(db_chunk)

    # 5. Embed + add to FAISS
    chunks_with_ids = [
        {
            'text': db_chunk.text,
            'chunk_db_id': db_chunk.id,
            'paper_id': paper_db_obj.id,
        }
        for db_chunk in db_chunks
    ]
    faiss_ids = add_chunks_to_index(chunks_with_ids)

    # 6. Store faiss_id back on each chunk
    for db_chunk, fid in zip(db_chunks, faiss_ids):
        db_chunk.faiss_id = fid
        db_chunk.save(update_fields=['faiss_id'])

    # 7. Mark paper as processed
    paper_db_obj.processed = True
    paper_db_obj.chunk_count = len(db_chunks)
    paper_db_obj.save()

    return len(db_chunks)


def answer_question(question: str, paper_ids: list[int], mode: str = 'qa') -> dict:
    """
    Full Q&A pipeline.
    Returns {answer, sources: [{text, page, paper_id}]}
    """
    from papers.models import Chunk

    results = search_index(question, paper_ids=paper_ids)
    if not results:
        return {'answer': 'No relevant content found.', 'sources': []}

    chunk_db_ids = [r['chunk_db_id'] for r in results]
    chunks = Chunk.objects.filter(id__in=chunk_db_ids).select_related('paper')

    chunk_map = {c.id: c for c in chunks}
    ordered_chunks = [chunk_map[cid] for cid in chunk_db_ids if cid in chunk_map]

    context = build_context([c.text for c in ordered_chunks])
    answer = ask_llm(question, context, mode=mode)

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