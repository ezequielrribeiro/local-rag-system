# AGENTS.md

## Source of Truth
- `specs/rag-spec-v1.md` — authoritative spec; all implementation must conform to it.

## Project Structure (Spec-Defined)
```
rag-system/
├── specs/rag-spec-v1.md
├── data/
│   ├── raw/{user,tech,support}/   # Input: .md, .pdf, .php files
│   ├── processed/                  # Serialized chunks
│   └── vector_db/                  # Persisted vector index
├── src/
│   ├── ingestion/{chunkers,pipeline}.py
│   ├── retrieval/{vector_store,router}.py
│   ├── generation/{prompts,llm_client}.py
│   ├── cli/repl.py                 # Interactive REPL
│   ├── clipboard/loader.py         # Clipboard read + chunk
│   └── api/{app,service,schemas,config}.py  # REST search API (FastAPI)
├── tests/{test_chunking,test_retrieval,test_api}.py
├── config.yaml
├── requirements.txt
├── requirements-dev.txt
├── main.py
├── README.md
└── LICENSE
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
mkdir data\raw\user data\raw\tech data\raw\support data\processed data\vector_db
```

## Setup (dev/testes)
```bash
pip install -r requirements-dev.txt   # inclui requirements.txt + pytest + httpx2
pytest tests/ -v
```

## Commands
```bash
python main.py ingest                                    # Index documents
python main.py query "sua pergunta"                      # Single query
python main.py query "sua pergunta" --doc-type tech      # Filter by domain
python main.py query "sua pergunta" --paste              # + paste clipboard context
python main.py query "sua pergunta" --clipboard          # Copy prompt to clipboard (no LLM)
python main.py serve --host 127.0.0.1 --port 8000        # REST API (search only, no LLM)
```

REST API endpoints: `GET /health`, `POST /api/search`. Shared vector store is cached per process.

REPL commands: `/help`, `/clear`, `/model [name]`, `/doc-type [mode]`, `/clip [query]`, `/paste [query]`, `/quit`, `/reset`.

## Architecture Constraints
- **Local-only** — no cloud APIs for embeddings or LLM inference. Use Ollama for local LLM.
- **Hybrid search** — BM25 + vector embeddings (e.g., `bge-m3` or `nomic-embed-text`).
- **Vendor exclusion** — `/vendor`, `/node_modules`, `/.venv`, `/venv`, `/__pycache__`, `/site-packages`, `/dist`, `/build`, `/.next`, `/coverage` paths and `*.min.js` / `*.bundle.js` / `*.pack.js` artifacts must be excluded during ingestion.
- **Code chunking** — PHP: `<?php` blocks + recursive split. Python: stdlib `ast` for `def`/`class` boundaries (regex fallback when the file does not parse). JavaScript: recursive split on declaration separators + regex symbol detection.
- **Domain routing** — keyword-based `doc_type` filtering per spec §4 QueryRouter rules.
- **Grounded generation** — LLM must refuse to answer if no relevant context found (Portuguese: `"Informação não encontrada na documentação técnica fornecida."`).

## Data Contract: DocumentChunk
```python
{
  "chunk_id": sha256(source_path + chunk_index),
  "page_content": str,
  "metadata": {
    "source": str, "filename": str,
    "doc_type": "user"|"tech"|"support",
    "format": "pdf"|"markdown"|"php_code"|"python_code"|"javascript_code",
    "chunk_index": int,
    "page_number": int | null,       # PDF only
    "headers": dict | null,          # Markdown only
    "detected_classes": list[str],   # Code files only
    "detected_functions": list[str], # Code files only
  }
}
```

## Testing (BDD Scenarios from Spec)
Run single test: `pytest tests/test_<name>.py -v`
1. **test_chunking** — PHP vendor exclusion, markdown heading splitting, Python AST chunking, JavaScript symbol detection
2. **test_retrieval** — hybrid search, metadata filtering, hallucination prevention, query routing for php/python/javascript
3. **test_api** — health, doc_type routing, serialization, no-index 503
