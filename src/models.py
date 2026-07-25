import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DocType(str, Enum):
    USER = "user"
    TECH = "tech"
    SUPPORT = "support"


class FileFormat(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    PHP_CODE = "php_code"


@dataclass
class ChunkMetadata:
    source: str
    filename: str
    doc_type: DocType
    format: FileFormat
    chunk_index: int
    page_number: Optional[int] = None
    headers: Optional[dict[str, str]] = None
    detected_classes: Optional[list[str]] = None
    detected_functions: Optional[list[str]] = None


@dataclass
class DocumentChunk:
    chunk_id: str
    page_content: str
    metadata: ChunkMetadata

    @staticmethod
    def generate_id(source: str, chunk_index: int) -> str:
        raw = f"{source}:{chunk_index}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
