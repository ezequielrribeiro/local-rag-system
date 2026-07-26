import logging

import pyperclip

from src.models import ChunkMetadata, DocType, DocumentChunk, FileFormat

logger = logging.getLogger(__name__)


def read_clipboard() -> str:
    try:
        text = pyperclip.paste()
        if not text or not text.strip():
            logger.warning("Clipboard is empty")
            return ""
        return text.strip()
    except Exception:
        logger.exception("Failed to read clipboard")
        return ""


def make_clipboard_chunk(text: str, doc_type: str = "tech") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=DocumentChunk.generate_id("clipboard://paste", 0),
        page_content=text,
        metadata=ChunkMetadata(
            source="clipboard://paste",
            filename="clipboard.txt",
            doc_type=DocType(doc_type),
            format=FileFormat.MARKDOWN,
            chunk_index=0,
        ),
    )
