# Sarvo RAG Chatbot

A FastAPI-based **RAG (Retrieval-Augmented Generation)** chatbot for Falcon Reality. It ingests PDF documents, embeds and stores them in a local vector database, and answers user questions by retrieving relevant document chunks and passing them to an LLM as context.

## What it does

1. Loads PDF documents (from a folder or via upload) and splits them into overlapping text chunks.
2. Generates embeddings for each chunk using OpenAI's embedding model.
3. Stores chunks + embeddings in a persistent ChromaDB vector store, keyed with a content hash so the same document is never re-processed twice.
4. On a chat request, embeds the user's question, retrieves the most relevant chunks from the vector store, and asks an OpenAI chat model to answer using that context (citing the source document).

## Features

- **PDF ingestion** — bulk-load all PDFs from a folder, or upload a single PDF via API.
- **Duplicate detection** — documents are fingerprinted with SHA-256 so re-adding the same file is a no-op.
- **Chunking with overlap** — text is split into overlapping word chunks to preserve context across boundaries.
- **Vector search** — semantic similarity search over document chunks using ChromaDB.
- **Context-grounded answers** — responses are generated from retrieved chunks, with source documents cited.
- **Knowledge base status/inspection endpoints** — check readiness, chunk count, and list ingested documents.
- **Web search tool (in progress)** — a DuckDuckGo-based search tool (`src/tools/websearch_tool.py`) is scaffolded for augmenting answers when the knowledge base lacks an answer, but is not yet wired into the chat endpoint.

## Tech stack / frameworks

| Purpose             | Library                              |
|----------------------|---------------------------------------|
| Web framework        | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| LLM / embeddings     | [OpenAI API](https://platform.openai.com/) (`gpt-4o-mini`, `text-embedding-3-small`) via the `openai` Python SDK |
| Vector database       | [ChromaDB](https://www.trychroma.com/) (persistent local storage) |
| PDF parsing           | `PyPDF2` |
| Config / secrets      | `python-dotenv` |
| Web search tool (scaffolded) | `ddgs` (DuckDuckGo Search), `strands` |
| Package/env management | [`uv`](https://github.com/astral-sh/uv) |

## Project structure

```
main.py                        # FastAPI app entrypoint, mounts the API router
src/
  endpoints/
    router.py                  # Top-level API router
    rag_chatbot.py              # Core RAG logic: PDF loading, chunking, embeddings,
                                 # ChromaDB vector store, chat endpoint
  tools/
    websearch_tool.py           # DuckDuckGo web search tool (not yet wired in)
Falcon_Reality_blog_Pdf/        # Source PDFs used to seed the knowledge base
vectordb_storage/               # Persistent ChromaDB storage (auto-created)
requirements.txt                # Pinned dependencies
pyproject.toml                  # Project metadata (Python >=3.11)
.env.example                    # Template for required environment variables
run.ps1                         # Convenience script to launch the server on Windows
```

## Setup

### Prerequisites

- Python 3.11 (pinned in `.python-version`)
- [`uv`](https://github.com/astral-sh/uv) installed (recommended), or `pip`
- An OpenAI API key

### 1. Create a virtual environment and install dependencies

Using `uv` (recommended):

```powershell
uv venv --python 3.11
uv pip install -r requirements.txt --python .venv
```

Or using plain `venv` + `pip`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example file and fill in your key:

```powershell
copy .env.example .env
```

`.env`:

```
OPENAI_API_KEY=your-openai-api-key-here
HOST=0.0.0.0
PORT=8000
```

Only `OPENAI_API_KEY` is currently read by the app (`src/endpoints/rag_chatbot.py`); `HOST`/`PORT` are provided for convenience if you wire them into the run command.

## Running the server

**Windows (PowerShell):**

```powershell
.\run.ps1
```

This sets `PYTHONUTF8=1` (avoids a `UnicodeEncodeError` from emoji in console log output on Windows' default codepage) and starts Uvicorn with auto-reload.

**Manually / other platforms:**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

## API endpoints

All endpoints are mounted under the `/api` prefix.

| Method | Path                | Description |
|--------|----------------------|--------------|
| GET    | `/`                   | Health check (root, outside `/api`) |
| POST   | `/api/chat`            | Ask a question; returns an AI answer grounded in the knowledge base plus cited sources |
| POST   | `/api/upload-document`  | Upload a single PDF to add to the knowledge base |
| POST   | `/api/add-documents`    | Bulk-process all new PDFs in a folder (default: `Falcon_Reality_blog_Pdf/`) |
| GET    | `/api/documents`        | List all documents currently in the knowledge base |
| GET    | `/api/status`           | Check knowledge base readiness and chunk count |

### Example: chat request

```json
POST /api/chat
{
  "message": "What is Virtual Reality?",
  "top_k": 3
}
```

```json
{
  "answer": "...",
  "sources": ["What is Virtual Reality Why should you care about it.pdf"],
  "success": true
}
```

## Notes

- The vector database persists to `./vectordb_storage/` on disk — delete this folder to reset the knowledge base from scratch.
- On first run, call `POST /api/add-documents` to ingest the PDFs in `Falcon_Reality_blog_Pdf/` before chatting.
