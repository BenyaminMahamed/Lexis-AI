# AI Research Assistant (RAG Pipeline)

A full-stack **Retrieval-Augmented Generation (RAG)** platform designed to facilitate deep interaction with academic literature. This project implements a sophisticated AI pipeline—from high-fidelity PDF parsing and semantic chunking to vector embedding storage and context-aware querying.

## 🚀 Key Features

*   **High-Fidelity PDF Ingestion:** Extracts and cleans text from academic PDFs using `PyMuPDF` (fitz).
*   **Semantic Vector Embeddings:** Leverages `sentence-transformers` (all-MiniLM-L6-v2) to map document context into high-dimensional space.
*   **Real-time Vector Search:** Implements `FAISS` (Facebook AI Similarity Search) for millisecond-latency similarity indexing and retrieval.
*   **Context-Grounded Q&A:** Synchronizes retrieved document context with Large Language Models (LLMs) to ensure hallucination-resistant, source-backed answers.
*   **Research Dashboard:** A robust Django-based management interface for multi-document analysis and persistent storage.

## 🛠️ Technical Stack

*   **Backend:** Python 3.13, Django 6.0
*   **AI/ML Stack:** PyTorch, Sentence-Transformers, FAISS, OpenAI/Gemini API
*   **Data Science:** NumPy, SciPy, Scikit-learn
*   **Database:** SQLite (Relational Metadata), FAISS Index (Vector Store)
*   **Frontend:** Django Templates, Bootstrap 5

## 📂 Project Structure

```text
├── config/              # Core Django configurations (Settings, WSGI, ASGI)
├── papers/
│   ├── models.py        # Relational schema for Papers and Text Chunks
│   ├── pipeline.py      # Core AI Engine (Embeddings, RAG Logic, Semantic Search)
│   ├── views.py         # App logic for document processing and Q&A endpoints
│   └── urls.py          # Application-level routing
├── media/               # Persistent storage for uploaded PDF assets
├── faiss_index/         # Serialized local vector database
└── manage.py            # Project execution entry point
```

## ⚙️ Installation & Setup
1. Clone the Repository
```text
PowerShell
git clone https://github.com/BenyaminMahamed/AI-Research-Assistant.git
cd AI-Research-Assistant
Environment Configuration
```
2. Environment Configuration
PowerShell
```text
python -m venv venv
.\venv\Scripts\activate
Install Dependencies
```
3. Install Dependencies
PowerShell
```text
pip install -r requirements.txt
Environment Variables
Create a .env file in the root directory:
```
4. Environment Variables
Create a .env file in the root directory:
Plaintext
```text
DEBUG=True
SECRET_KEY=your_django_secret_key
OPENAI_API_KEY=your_api_key_here
Initialize Database & Launch
```
5. Initialize Database & Launch
PowerShell
```text
PowerShell
python manage.py makemigrations papers
python manage.py migrate
python manage.py runserver
```
## 🧠 Architectural Workflow (The RAG Pipeline)
Ingestion & Preprocessing: PDFs are parsed and divided into overlapping semantic chunks to preserve context across boundaries.

Vectorization: Each chunk is passed through a transformer model to generate a unique numerical embedding.

Indexing: These embeddings are indexed via FAISS, creating a searchable mathematical map of the document's knowledge.

Retrieval: User queries are embedded in real-time; the system performs a "nearest neighbor" search to find the most relevant document sections.

Augmented Generation: The retrieved context is fed into the LLM as a "grounding truth," ensuring generated answers are strictly based on the uploaded research.

## 👨‍💻 Author
Benyamin Mahamed

BSc Computer Science @ University of Westminster (Predicted First-Class)
