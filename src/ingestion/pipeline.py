import json
import logging
import os
from dataclasses import asdict

from src.ingestion.chunkers import (
    BaseChunker,
    JavaScriptChunker,
    MarkdownChunker,
    PDFChunker,
    PHPChunker,
    PythonChunker,
)
from src.models import DocType, DocumentChunk

logger = logging.getLogger(__name__)

DOC_TYPE_MAP: dict[str, DocType] = {
    "user": DocType.USER,
    "tech": DocType.TECH,
    "support": DocType.SUPPORT,
}

EXTENSION_MAP: dict[str, type[BaseChunker]] = {
    ".php": PHPChunker,
    ".py": PythonChunker,
    ".js": JavaScriptChunker,
    ".md": MarkdownChunker,
    ".pdf": PDFChunker,
}

SOURCE_KIND_BY_DOC_TYPE = {
    DocType.USER: "manual",
    DocType.TECH: "code",
    DocType.SUPPORT: "ticket",
}


def serialize_chunks(chunks: list[DocumentChunk]) -> list[dict]:
    result = []
    for chunk in chunks:
        d = asdict(chunk)
        d["metadata"]["doc_type"] = d["metadata"]["doc_type"].value
        d["metadata"]["format"] = d["metadata"]["format"].value
        result.append(d)
    return result


# Compatibilidade com o nome anterior.
_serialize_chunks = serialize_chunks


def _source_kind(file_path: str, doc_type: DocType) -> str | None:
    normalized = file_path.replace("\\", "/").lower()
    if "/_harness/db/" in normalized:
        return "db_schema"
    if "/_harness/links/" in normalized:
        return "db_links"
    if "/_harness/code/" in normalized:
        return "code"
    return SOURCE_KIND_BY_DOC_TYPE.get(doc_type)


def iter_source_files(raw_dir: str):
    """Percorre os arquivos indexáveis e devolve (path, doc_type, chunker_cls)."""
    if not os.path.isdir(raw_dir):
        return

    for dir_entry in os.scandir(raw_dir):
        if not dir_entry.is_dir():
            continue

        doc_type = DOC_TYPE_MAP.get(dir_entry.name)
        if doc_type is None:
            logger.warning("Unknown directory %s, skipping", dir_entry.name)
            continue

        for root, _dirs, files in os.walk(dir_entry.path):
            for filename in files:
                file_path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()
                chunker_cls = EXTENSION_MAP.get(ext)
                if chunker_cls is None:
                    continue
                yield file_path, doc_type, chunker_cls


def _build_chunker(
    chunker_cls: type[BaseChunker],
    chunk_size: int,
    chunk_overlap: int,
    table_max_size: int,
) -> BaseChunker:
    if chunker_cls is MarkdownChunker:
        return chunker_cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            table_max_size=table_max_size,
        )
    return chunker_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_file(
    file_path: str,
    doc_type: DocType,
    chunker_cls: type[BaseChunker],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    table_max_size: int = 4000,
) -> list[DocumentChunk]:
    chunker = _build_chunker(chunker_cls, chunk_size, chunk_overlap, table_max_size)

    if hasattr(chunker, "should_exclude") and chunker.should_exclude(file_path):
        return []

    chunks = chunker.chunk(file_path, doc_type)
    source_kind = _source_kind(file_path, doc_type)
    for chunk in chunks:
        chunk.metadata.source_kind = source_kind
    return chunks


def run_ingestion(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    table_max_size: int = 4000,
) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    stats: dict[str, int] = {"files_processed": 0, "files_skipped": 0, "chunks": 0}

    for file_path, doc_type, chunker_cls in iter_source_files(raw_dir):
        try:
            chunks = chunk_file(
                file_path,
                doc_type,
                chunker_cls,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                table_max_size=table_max_size,
            )
            if not chunks:
                stats["files_skipped"] += 1
                continue

            all_chunks.extend(chunks)
            stats["files_processed"] += 1
            stats["chunks"] += len(chunks)
            logger.info("Processed %s -> %d chunks", file_path, len(chunks))
        except Exception:
            logger.exception("Failed to process %s", file_path)
            stats["files_skipped"] += 1

    _write_chunks_json(processed_dir, all_chunks)

    logger.info(
        "Ingestion complete: %d files processed, %d skipped, %d chunks written to %s",
        stats["files_processed"],
        stats["files_skipped"],
        stats["chunks"],
        os.path.join(processed_dir, "chunks.json"),
    )

    return all_chunks


def run_ingestion_for_files(
    file_paths: list[str],
    raw_dir: str = "data/raw",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    table_max_size: int = 4000,
) -> list[DocumentChunk]:
    """Processa apenas os arquivos informados (ingestão incremental)."""
    mapping = {
        path: (doc_type, chunker_cls)
        for path, doc_type, chunker_cls in iter_source_files(raw_dir)
    }

    chunks: list[DocumentChunk] = []
    for path in file_paths:
        entry = mapping.get(path)
        if entry is None:
            continue
        doc_type, chunker_cls = entry
        try:
            chunks.extend(
                chunk_file(
                    path,
                    doc_type,
                    chunker_cls,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    table_max_size=table_max_size,
                )
            )
        except Exception:
            logger.exception("Failed to process %s", path)

    return chunks


def _write_chunks_json(processed_dir: str, chunks: list[DocumentChunk]) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialize_chunks(chunks), f, ensure_ascii=False, indent=2)
