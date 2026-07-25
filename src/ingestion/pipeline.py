import json
import logging
import os
from dataclasses import asdict

from src.ingestion.chunkers import (
    BaseChunker,
    MarkdownChunker,
    PDFChunker,
    PHPChunker,
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
    ".md": MarkdownChunker,
    ".pdf": PDFChunker,
}


def _serialize_chunks(chunks: list[DocumentChunk]) -> list[dict]:
    result = []
    for chunk in chunks:
        d = asdict(chunk)
        d["metadata"]["doc_type"] = d["metadata"]["doc_type"].value
        d["metadata"]["format"] = d["metadata"]["format"].value
        result.append(d)
    return result


def run_ingestion(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    stats: dict[str, int] = {"files_processed": 0, "files_skipped": 0, "chunks": 0}

    for dir_entry in os.scandir(raw_dir):
        if not dir_entry.is_dir():
            continue

        dir_name = dir_entry.name
        doc_type = DOC_TYPE_MAP.get(dir_name)
        if doc_type is None:
            logger.warning("Unknown directory %s, skipping", dir_name)
            continue

        for root, _dirs, files in os.walk(dir_entry.path):
            for filename in files:
                file_path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()
                chunker_cls = EXTENSION_MAP.get(ext)

                if chunker_cls is None:
                    stats["files_skipped"] += 1
                    continue

                chunker = chunker_cls()
                if hasattr(chunker, "should_exclude") and chunker.should_exclude(
                    file_path
                ):
                    stats["files_skipped"] += 1
                    continue

                try:
                    chunks = chunker.chunk(file_path, doc_type)
                    all_chunks.extend(chunks)
                    stats["files_processed"] += 1
                    stats["chunks"] += len(chunks)
                    logger.info(
                        "Processed %s -> %d chunks", file_path, len(chunks)
                    )
                except Exception:
                    logger.exception("Failed to process %s", file_path)
                    stats["files_skipped"] += 1

    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "chunks.json")
    serialized = _serialize_chunks(all_chunks)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)

    logger.info(
        "Ingestion complete: %d files processed, %d skipped, %d chunks written to %s",
        stats["files_processed"],
        stats["files_skipped"],
        stats["chunks"],
        output_path,
    )

    return all_chunks
