# Local Multi-Layer RAG System

RAG system local para documentação de usuário, código PHP legado e tickets de suporte. Embeddings com `bge-m3`, busca híbrida (BM25 + vetorial) e inferência via Ollama.

## Requisitos

- Python 3.13+
- [Ollama](https://ollama.com) com modelo baixado (ex: `ollama pull llama3.2`)

O modelo de embedding `bge-m3` (`sentence-transformers`) é baixado automaticamente no primeiro `ingest`.

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Para rodar os testes, instale também as dependências de desenvolvimento:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Crie os diretórios de dados antes do primeiro uso:

```bash
mkdir data\raw\user data\raw\tech data\raw\support data\processed data\vector_db
```

## Uso

### 1. Coloque documentos em `data/raw/`

```
data/raw/
├── user/       # PDFs e .md de documentação do usuário
├── tech/       # Código PHP e wikis técnicas (.md)
└── support/    # Tickets de suporte (.pdf,.md)
```

### 2. Ingestão

Processa, chunkifica e indexa todos os documentos:

```bash
python main.py ingest
```

### 3. Consultas

**Modo interativo (REPL):**

```bash
python main.py chat
```

Comandos disponíveis no REPL:

| Comando | Descrição |
|---|---|
| `/help` | Lista comandos |
| `/clear` | Limpa a tela |
| `/model` | Mostra modelo atual |
| `/model <nome>` | Troca modelo (ex: `/model llama3.2`) |
| `/doc-type` | Mostra filtro de domínio |
| `/doc-type <modo>` | Define filtro: `auto`, `user`, `tech`, `support` |
| `/paste` | Lê clipboard como contexto extra |
| `/paste <query>` | Lê clipboard e executa query |
| `/clip` | Copia prompt completo ao clipboard (sem chamar LLM) |
| `/clip <query>` | Copia prompt completo ao clipboard (sem chamar LLM) |
| `/quit` ou `/exit` | Sai do modo interativo (libera RAM) |
| `/reset` | Limpa RAM + apaga banco vetorial e chunks do disco |

**Consulta única:**

```bash
python main.py query "Como alterar a senha?"
python main.py query "Função de login no PHP" --doc-type tech
python main.py query "O que esse código faz?" --paste
python main.py query "Explique isso" --clipboard          # Copia prompt ao clipboard
```

### 4. API REST (busca)

Serve a busca híbrida via HTTP, sem chamar o LLM (apenas recuperação de chunks). Docs
interativas (Swagger) ficam em `http://127.0.0.1:8000/docs`.

```bash
python main.py serve                              # 127.0.0.1:8000
python main.py serve --host 127.0.0.1 --port 8080
```

**Endpoints:**

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status do índice (carregado? nº de chunks) |
| `POST` | `/api/search` | Busca híbrida; retorna chunks recuperados (sem LLM) |

**Exemplos:**

```bash
# Health check
curl -s http://127.0.0.1:8000/health

# Busca simples (rota de domínio automática por keywords)
curl -s -X POST http://127.0.0.1:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Função de login no PHP"}'

# Filtro explícito por domínio
curl -s -X POST http://127.0.0.1:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Como alterar a senha?","doc_type":"user"}'

# Definir top_k
curl -s -X POST http://127.0.0.1:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"bug no checkout","top_k":10}'
```

**Corpo de resposta** (`POST /api/search`):

```json
{
  "query": "Função de login no PHP",
  "doc_type_used": "tech",
  "routed_to": ["tech", "support"],
  "count": 5,
  "results": [
    {
      "chunk_id": "...",
      "page_content": "...",
      "metadata": {
        "source": "...", "filename": "...", "doc_type": "tech",
        "format": "php_code", "chunk_index": 0,
        "detected_functions": ["login"]
      }
    }
  ]
}
```

`doc_type` aceita `auto` (padrão), `user`, `tech`, `support`. Se não houver um índice
(`data/vector_db/`), a API responde `503` — rode `python main.py ingest` primeiro.
Requisições com `query` vazia respondem `422`.

## Estrutura

```
src/
├── models.py              # Contrato DocumentChunk
├── ingestion/
│   ├── chunkers.py        # Chunkers: PHP, Markdown, PDF
│   └── pipeline.py        # Orquestrador de ingestão
├── retrieval/
│   ├── vector_store.py    # Busca híbrida (bge-m3 + BM25)
│   └── router.py          # Roteamento por keywords
├── generation/
│   ├── prompts.py         # Templates de prompt por domínio
│   └── llm_client.py      # Cliente Ollama (HTTP + streaming)
├── cli/
│   └── repl.py            # Modo interativo (prompt_toolkit + rich)
├── clipboard/
│   └── loader.py          # Leitura e chunk do clipboard
└── api/
    ├── app.py             # Factory FastAPI (app + rotas)
    ├── service.py         # RAGSearchService (cache do vector store)
    ├── schemas.py         # Models Pydantic de busca
    └── config.py          # Carregamento de config com cache
```

## Gerenciamento de Memória

| Ação | O que libera |
|---|---|
| `/quit`, `/exit` ou `Ctrl+D` | Libera da RAM: modelo `bge-m3`, embeddings, índice BM25, chunks |
| `/reset` | Libera RAM + apaga `data/vector_db/` e `data/processed/chunks.json` do disco |

Após `/reset`, execute `python main.py ingest` para reindexar os documentos.

## Configuração

Edite `config.yaml` para alterar modelo de embedding, modelo LLM, `top_k`, etc.

O bloco `server` define os valores padrão da API REST (sobrescritos por `--host`/`--port` na linha de comando):

```yaml
server:
  host: 127.0.0.1
  port: 8000
```
