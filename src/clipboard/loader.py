import logging

import pyperclip

from src.generation.prompts import PROMPT_TEMPLATE, get_system_prompt
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


def copy_to_clipboard(text: str) -> bool:
    try:
        pyperclip.copy(text)
        return True
    except Exception:
        logger.exception("Failed to copy to clipboard")
        return False


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


def build_full_prompt(
    query: str,
    chunks: list[DocumentChunk],
    doc_type: str = "tech",
) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.source
        filename = chunk.metadata.filename
        context_parts.append(
            f"[{i}] Source: {source} | File: {filename}\n{chunk.page_content}"
        )
    retrieved_context = "\n\n---\n\n".join(context_parts)
    user_prompt = PROMPT_TEMPLATE.format(
        retrieved_context=retrieved_context,
        user_query=query,
    )
    system_prompt = get_system_prompt(doc_type)
    return f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
