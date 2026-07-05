# Lexis — AI Research Assistant

A full-stack **Retrieval-Augmented Generation (RAG)** platform for deep interaction with academic literature. Upload PDFs, ask questions, generate structured summaries, critique arguments, and compare papers — all grounded in the actual document content.

Built as a portfolio project demonstrating end-to-end AI/ML engineering: PDF ingestion, semantic chunking, vector embeddings, FAISS similarity search, and LLM-powered generation.

---

## Features

- **PDF Ingestion** — High-fidelity text extraction via PyMuPDF with page-level metadata preserved
- **Semantic Chunking** — Overlapping word-window chunking to maintain context across boundaries
- **Vector Embeddings** — Sentence-level embeddings using `all-MiniLM-L6-v2` (384-dimensional)
- **FAISS Vector Search** — Millisecond-latency nearest-neighbour retrieval across indexed chunks
- **Four Analysis Modes** — Q&A, Summarise, Critique, and Compare across multiple papers
- **Source Attribution** — Every answer surfaces the exact chunks and page numbers it was grounded in
- **Demo Mode** — Fully functional RAG pipeline without an API key; add `GEMINI_API_KEY` to enable LLM responses

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.13, Django 6.0 |
| AI/ML | Sentence-Transformers, FAISS, Google Gemini 2.0 Flash |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace |
| PDF Parsing | PyMuPDF (fitz) |
| Database | SQLite (metadata), FAISS index (vectors) |
| Frontend | Django Templates (custom dark academic UI) |
| API | Django REST Framework |

---

## Project Structure

```
├── config/              # Django settings, URLs, WSGI/ASGI
├── papers/
│   ├── migrations/      # Database migrations
│   ├── models.py        # Paper and Chunk schema
│   ├── pipeline.py      # RAG engine: extraction, chunking, embeddings, FAISS, Gemini
│   ├── views.py         # Request handling and API endpoints
│   └── urls.py          # URL routing
├── templates/
│   └── papers/          # base.html, index.html, detail.html
├── .env.example         # Environment variable template
├── manage.py
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/BenyaminMahamed/AI-Research-Assistant.git
cd AI-Research-Assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
DJANGO_SECRET_KEY=your_django_secret_key_here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
GEMINI_API_KEY=your_gemini_api_key_here
```

Generate a Django secret key:

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com). The app runs in demo mode without one — the RAG pipeline still works, only LLM generation is disabled.

### 5. Initialise the database and run

```bash
python manage.py migrate
python manage.py runserver
```

Visit `http://127.0.0.1:8000`.

---

## RAG Pipeline

```
PDF Upload
    │
    ▼
Text Extraction (PyMuPDF, page-level)
    │
    ▼
Semantic Chunking (500-word windows, 50-word overlap)
    │
    ▼
Embedding Generation (all-MiniLM-L6-v2, 384-dim, normalised)
    │
    ▼
FAISS Indexing (IndexFlatIP — inner product similarity)
    │
    ▼
Query → Embed → Nearest-Neighbour Search → Retrieve Top-K Chunks
    │
    ▼
Context + Question → Gemini 2.0 Flash → Grounded Answer + Sources
```

---

## Author

**Benyamin Mahamed**  
BSc (Hons) Computer Science, University of Westminster  
[GitHub](https://github.com/BenyaminMahamed)
