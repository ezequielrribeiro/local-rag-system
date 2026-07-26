# Local Multi-Layer RAG System

RAG system local para documentação de usuário, código PHP legado e tickets de suporte. Embeddings com `bge-m3`, busca híbrida (BM25 + vetorial) e inferência via Ollama.

## Requisitos

- Python 3.13+
- [Ollama](https://ollama.com) com modelo baixado (ex: `ollama pull llama3.2`)

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
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
| `/quit` ou `/exit` | Sai do modo interativo |

**Consulta única:**

```bash
python main.py query "Como alterar a senha?"
python main.py query "Função de login no PHP" --doc-type tech
```

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
└── cli/
    └── repl.py            # Modo interativo (prompt_toolkit + rich)
```

## Configuração

Edite `config.yaml` para alterar modelo de embedding, modelo LLM, `top_k`, etc.
