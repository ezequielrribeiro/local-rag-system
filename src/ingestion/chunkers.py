import os
import re
from abc import ABC, abstractmethod

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

from src.models import ChunkMetadata, DocType, DocumentChunk, FileFormat


class BaseChunker(ABC):
    VENDOR_EXCLUDE_PATTERNS = [
        re.compile(r"[/\\]vendor[/\\]"),
        re.compile(r"[/\\]node_modules[/\\]"),
    ]

    def should_exclude(self, file_path: str) -> bool:
        for pattern in self.VENDOR_EXCLUDE_PATTERNS:
            if pattern.search(file_path):
                return True
        return False

    @abstractmethod
    def chunk(self, file_path: str, doc_type: DocType) -> list[DocumentChunk]:
        ...


class PHPChunker(BaseChunker):
    _CLASS_REGEX = re.compile(r"(?:^|\n)\s*(?:abstract\s+)?(?:final\s+)?class\s+(\w+)")
    _FUNCTION_REGEX = re.compile(
        r"(?:^|\n)\s*(?:public|protected|private|static|\s)*\s*function\s+(\w+)\s*\("
    )

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\nclass ",
                "\nfunction ",
                "\npublic function ",
                "\nprotected function ",
                "\nprivate function ",
                "\n\n",
                "\n",
                ".",
            ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _detect_classes(self, text: str) -> list[str]:
        return self._CLASS_REGEX.findall(text)

    def _detect_functions(self, text: str) -> list[str]:
        return self._FUNCTION_REGEX.findall(text)

    def chunk(self, file_path: str, doc_type: DocType) -> list[DocumentChunk]:
        if self.should_exclude(file_path):
            return []

        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()

        full_classes = self._detect_classes(text)
        full_functions = self._detect_functions(text)

        langchain_docs = self.splitter.create_documents([text])
        chunks = []
        for i, lc_doc in enumerate(langchain_docs):
            chunk_text = lc_doc.page_content
            chunk_classes = self._CLASS_REGEX.findall(chunk_text)
            chunk_functions = self._FUNCTION_REGEX.findall(chunk_text)
            metadata = ChunkMetadata(
                source=file_path,
                filename=os.path.basename(file_path),
                doc_type=doc_type,
                format=FileFormat.PHP_CODE,
                chunk_index=i,
                detected_classes=chunk_classes or (full_classes if i == 0 else None),
                detected_functions=chunk_functions
                or (full_functions if i == 0 else None),
            )
            chunk = DocumentChunk(
                chunk_id=DocumentChunk.generate_id(file_path, i),
                page_content=chunk_text,
                metadata=metadata,
            )
            chunks.append(chunk)
        return chunks


class MarkdownChunker(BaseChunker):
    def __init__(self):
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )

    def chunk(self, file_path: str, doc_type: DocType) -> list[DocumentChunk]:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()

        langchain_docs = self.splitter.split_text(text)
        chunks = []
        for i, lc_doc in enumerate(langchain_docs):
            metadata = ChunkMetadata(
                source=file_path,
                filename=os.path.basename(file_path),
                doc_type=doc_type,
                format=FileFormat.MARKDOWN,
                chunk_index=i,
                headers=lc_doc.metadata if lc_doc.metadata else None,
            )
            chunk = DocumentChunk(
                chunk_id=DocumentChunk.generate_id(file_path, i),
                page_content=lc_doc.page_content,
                metadata=metadata,
            )
            chunks.append(chunk)
        return chunks


class PDFChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, file_path: str, doc_type: DocType) -> list[DocumentChunk]:
        reader = PdfReader(file_path)
        chunks = []
        chunk_index = 0
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text.strip():
                continue

            page_chunks = self.splitter.split_text(text)
            for page_chunk in page_chunks:
                metadata = ChunkMetadata(
                    source=file_path,
                    filename=os.path.basename(file_path),
                    doc_type=doc_type,
                    format=FileFormat.PDF,
                    chunk_index=chunk_index,
                    page_number=page_num,
                )
                chunk = DocumentChunk(
                    chunk_id=DocumentChunk.generate_id(file_path, chunk_index),
                    page_content=page_chunk,
                    metadata=metadata,
                )
                chunks.append(chunk)
                chunk_index += 1
        return chunks
