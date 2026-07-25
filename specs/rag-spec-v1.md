# Spec: Local Multi-Layer RAG System (`rag-spec-v1`)

```yaml
version: 1.0.0
title: Local Spec-Driven RAG for Legacy PHP & Multi-Source Support
author: System Architect
status: DRAFT
date: 2026-07-25
architecture_pattern: Spec-Driven Modular RAG
execution_environment: Local (Air-Gapped / Privacy-First)
```

---

## 1. Overview & System Purpose

Design and implement a fully local, privacy-compliant Retrieval-Augmented Generation (RAG) system tailored for individual developer use. The system aggregates, chunks, indexes, and queries three distinct domain knowledge layers:

1. **User Documentation:** End-user manuals (PDFs) and FAQs (Markdown).
2. **Technical Documentation:** Legacy PHP source code and Wiki documentation (Markdown).
3. **Support Documentation:** Customer support tickets (PDFs) and issue resolution write-ups (Markdown).

The architecture enforces strict metadata tagging, hybrid search (Lexical + Vector), and prompt isolation per knowledge domain.

---

## 2. Requirements Specification

### 2.1 Functional Requirements

* **[REQ-F01] Multi-Source Ingestion Engine:** The system MUST parse `.md`, `.pdf`, and `.php` files from structured directories without cross-contaminating document contexts.
* **[REQ-F02] Domain-Aware Metadata Assignment:** Every generated chunk MUST include mandatory metadata keys: `source_path`, `filename`, `doc_type` (`user` | `tech` | `support`), `format`, and `timestamp`.
* **[REQ-F03] Specialized Language Chunking:** PHP source code MUST be split using language-aware boundaries (classes, functions, methods) while automatically excluding third-party vendor directories (`/vendor`, `/node_modules`).
* **[REQ-F04] Structural Markdown Chunking:** Markdown files MUST be split along heading hierarchies (`#`, `##`, `###`) to preserve context.
* **[REQ-F05] Hybrid Retrieval Mechanism:** The retrieval system MUST perform dense vector search (embeddings) combined with sparse lexical search (BM25) over the vector index.
* **[REQ-F06] Domain Query Routing:** The query handler MUST route or filter searches based on user intent (e.g., technical code questions filter for `doc_type: ["tech", "support"]`).
* **[REQ-F07] Grounded Answer Generation:** The LLM prompt MUST mandate that answers strictly reference provided retrieved context and explicitly decline to answer if context is absent.

### 2.2 Non-Functional Requirements

* **[REQ-N01] Local Execution:** All embeddings, vector storage, and LLM inference MUST run locally without external cloud API calls.
* **[REQ-N02] Retrieval Latency:** Total retrieval phase (BM25 + Vector + Re-rank) MUST execute in under 800ms for collections up to 50,000 chunks.
* **[REQ-N03] Extensibility:** The ingestion and chunking logic MUST be decoupled from the vector database backend (VectorStore interface abstraction).

---

## 3. Data Architecture & Data Contracts

### 3.1 Input Directory Specification

```text
data/
└── raw/
    ├── user/     # [doc_type: user] PDFs, FAQ .md files
    ├── tech/     # [doc_type: tech] System Wiki .md, PHP source code
    └── support/  # [doc_type: support] Ticket .pdf files, solution .md files
```

### 3.2 Schema Definition: `DocumentChunk`

```typescript
interface DocumentChunk {
  chunk_id: string;          // Hash UUID (SHA-256 of source_path + chunk_index)
  page_content: string;      // Extracted chunk text
  metadata: {
    source: string;          // Absolute or relative file path
    filename: string;        // E.g., "Authentication.php"
    doc_type: "user" | "tech" | "support";
    format: "pdf" | "markdown" | "php_code";
    chunk_index: number;
    // Format-specific metadata
    page_number?: number;             // For PDF files
    headers?: Record<string, string>; // For Markdown (e.g., {"Header 1": "Overview"})
    detected_classes?: string[];     // For PHP files
    detected_functions?: string[];   // For PHP files
  };
}
```

---

## 4. Component Specification

### Component 1: `IngestionPipeline`
* **Inputs:** Path to `./data/raw`
* **Responsibility:** Traverse directories, instantiate appropriate chunkers based on file extensions, apply metadata tags, and output a validated list of `DocumentChunk` objects.
* **Dependencies:** `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter` (PHP/PDF), `pypdf`.

### Component 2: `HybridVectorStore`
* **Inputs:** `DocumentChunk[]`
* **Responsibility:** Generate vector embeddings locally (e.g., `bge-m3` or `nomic-embed-text`), create sparse BM25 index, and save index state to disk (`./data/vector_db/`).
* **Operations:**
  * `add_documents(chunks: DocumentChunk[]): void`
  * `hybrid_search(query: string, top_k: int, filter_metadata: dict): DocumentChunk[]`

### Component 3: `QueryRouter`
* **Inputs:** User query string
* **Responsibility:** Classify query domain or apply pre-filtering logic before executing retrieval.
  * **Rules:**
    * Keywords like "código", "função", "banco", "bug", "PHP" -> Filter: `doc_type in ["tech", "support"]`
    * Keywords like "como usar", "manual", "passo a passo", "login" -> Filter: `doc_type in ["user", "support"]`

### Component 4: `RAGGenerator`
* **Inputs:** User query + Top-K Chunks from `HybridVectorStore`
* **Responsibility:** Construct context-augmented prompt, execute local LLM inference (via Ollama or local endpoint), and append source citations.

---

## 5. System Prompts Specification

### System Prompt: Technical & Developer Context (`doc_type: tech | support`)

```text
You are an expert Lead Software Engineer specializing in legacy PHP web applications. 
Your primary task is to help a developer inspect, debug, and understand the legacy codebase and system architecture.

CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided code chunks and technical documentation below.
2. When referencing code or classes, specify the file name and function/method names if available in the context.
3. If the provided context does not contain enough information to answer the question, state clearly: "Informação não encontrada na documentação técnica fornecida."
4. Always write clean code snippets when proposing fixes or refactoring steps based on the legacy system rules.

CONTEXT:
---
{retrieved_context}
---

DEVELOPER QUERY: {user_query}
ANSWER:
```

---

## 6. Behavior-Driven Development (BDD) Test Scenarios

### Scenario 1: PHP Chunking excludes vendor dependencies
```gherkin
Feature: Specialized PHP Chunking
  Scenario: Ingesting technical codebase with third-party libraries
    Given a directory "data/raw/tech/vendor/guzzlehttp" containing PHP files
    And a file "data/raw/tech/controllers/UserController.php" containing legacy PHP code
    When the IngestionPipeline processes "data/raw/tech"
    Then no chunks should have "source" paths containing "/vendor/"
    And the generated chunks for "UserController.php" must contain metadata format "php_code"
```

### Scenario 2: Metadata Filtering during Hybrid Search
```gherkin
Feature: Hybrid Search Pre-Filtering
  Scenario: User asks a functional user-interface question
    Given an indexed database containing user manuals, PHP files, and support tickets
    When a user query "Como alterar a senha na tela de perfil?" is received
    And the QueryRouter identifies intent as "user_guidance"
    Then the retrieval filter MUST restrict search to "doc_type: ['user', 'support']"
    And retrieved chunks must NOT contain pure PHP source code chunks
```

### Scenario 3: Hallucination Prevention on Missing Context
```gherkin
Feature: Grounded Answer Generation
  Scenario: Querying about an unindexed feature
    Given an indexed database with no information about "Módulo de Pix"
    When the user asks "Como funciona o checkout com Pix no sistema?"
    And retrieval returns 0 relevant chunks above threshold
    Then the RAGGenerator response MUST contain "Informação não encontrada na documentação"
```

---

## 7. Implementation File Tree

```text
rag-system/
├── specs/
│   └── rag-spec.v1.md         # Esta especificação
├── data/
│   ├── raw/                  # Diretório de arquivos de entrada
│   ├── processed/            # Serialização dos chunks
│   └── vector_db/            # Banco vetorial persistido localmente
├── src/
│   ├── ingestion/
│   │   ├── chunkers.py       # Implem. do script de chunking especializado
│   │   └── pipeline.py       # Orquestrador de ingestão
│   ├── retrieval/
│   │   ├── vector_store.py   # Wrapper para BM25 + Embeddings
│   │   └── router.py         # Lógica de roteamento e filtro
│   └── generation/
│       ├── prompts.py        # Prompts do sistema por domínio
│       └── llm_client.py     # Conector para Ollama / LLM local
├── tests/
│   ├── test_chunking.py      # Testes unitários para PHP e MD
│   └── test_retrieval.py     # Testes de relevância da busca
├── config.yaml               # Modelos, top_k e parâmetros
└── main.py                   # CLI e interface principal
```
