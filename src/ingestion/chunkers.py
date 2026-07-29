import os
import re
from abc import ABC, abstractmethod

from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    _PHP_OPEN_RE = re.compile(r"<\?(?:php)?\s*")
    _PHP_CLOSE_RE = re.compile(r"\?>")

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self._php_splitter = RecursiveCharacterTextSplitter(
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
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _detect_classes(self, text: str) -> list[str]:
        return self._CLASS_REGEX.findall(text)

    def _detect_functions(self, text: str) -> list[str]:
        return self._FUNCTION_REGEX.findall(text)

    @staticmethod
    def _split_blocks(text: str) -> list[tuple[bool, str]]:
        blocks: list[tuple[bool, str]] = []
        pos = 0
        in_php = False
        while pos < len(text):
            if not in_php:
                m = PHPChunker._PHP_OPEN_RE.search(text, pos)
                if m is None:
                    blocks.append((False, text[pos:]))
                    break
                if m.start() > pos:
                    blocks.append((False, text[pos : m.start()]))
                blocks.append((True, m.group(0)))
                pos = m.end()
                in_php = True
            else:
                m = PHPChunker._PHP_CLOSE_RE.search(text, pos)
                if m is None:
                    blocks[-1] = (True, blocks[-1][1] + text[pos:])
                    break
                blocks[-1] = (True, blocks[-1][1] + text[pos : m.end()])
                pos = m.end()
                in_php = False
        return blocks

    def _chunk_php_block(
        self,
        php_code: str,
        file_path: str,
        filename: str,
        doc_type: DocType,
        chunk_index: int,
    ) -> tuple[list[DocumentChunk], int]:
        full_classes = self._detect_classes(php_code)
        full_functions = self._detect_functions(php_code)
        langchain_docs = self._php_splitter.create_documents([php_code])
        chunks = []
        for i, lc_doc in enumerate(langchain_docs):
            chunk_text = lc_doc.page_content
            chunk_classes = self._CLASS_REGEX.findall(chunk_text)
            chunk_functions = self._FUNCTION_REGEX.findall(chunk_text)
            metadata = ChunkMetadata(
                source=file_path,
                filename=filename,
                doc_type=doc_type,
                format=FileFormat.PHP_CODE,
                chunk_index=chunk_index,
                detected_classes=chunk_classes or (full_classes if i == 0 else None),
                detected_functions=chunk_functions
                or (full_functions if i == 0 else None),
            )
            chunks.append(DocumentChunk(
                chunk_id=DocumentChunk.generate_id(file_path, chunk_index),
                page_content=chunk_text,
                metadata=metadata,
            ))
            chunk_index += 1
        return chunks, chunk_index

    def _chunk_text_block(
        self,
        text: str,
        file_path: str,
        filename: str,
        doc_type: DocType,
        chunk_index: int,
    ) -> tuple[list[DocumentChunk], int]:
        sub_splits = self._text_splitter.split_text(text)
        chunks = []
        for sub in sub_splits:
            if not sub.strip():
                continue
            metadata = ChunkMetadata(
                source=file_path,
                filename=filename,
                doc_type=doc_type,
                format=FileFormat.MARKDOWN,
                chunk_index=chunk_index,
            )
            chunks.append(DocumentChunk(
                chunk_id=DocumentChunk.generate_id(file_path, chunk_index),
                page_content=sub,
                metadata=metadata,
            ))
            chunk_index += 1
        return chunks, chunk_index

    def chunk(self, file_path: str, doc_type: DocType) -> list[DocumentChunk]:
        if self.should_exclude(file_path):
            return []

        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()

        blocks = self._split_blocks(text)
        filename = os.path.basename(file_path)
        all_chunks: list[DocumentChunk] = []
        chunk_index = 0

        for is_php, content in blocks:
            content = content.strip()
            if not content:
                continue
            if is_php:
                chunks, chunk_index = self._chunk_php_block(
                    content, file_path, filename, doc_type, chunk_index
                )
            else:
                chunks, chunk_index = self._chunk_text_block(
                    content, file_path, filename, doc_type, chunk_index
                )
            all_chunks.extend(chunks)

        return all_chunks


class MarkdownChunker(BaseChunker):
    _HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    _HEADER_LEVELS = {"#": "Header 1", "##": "Header 2", "###": "Header 3"}
    _HEADER_DEPTH = {"#": 1, "##": 2, "###": 3}
    _LEVEL_MAP = {"Header 1": 1, "Header 2": 2, "Header 3": 3}

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def _parse_sections(text: str) -> list[tuple[dict[str, str], str]]:
        lines = text.splitlines(keepends=True)
        sections: list[tuple[dict[str, str], str]] = []
        active_headers: dict[str, str] = {}
        current_lines: list[str] = []

        def flush():
            if current_lines:
                content = "".join(current_lines).strip()
                if content:
                    sections.append((dict(active_headers), content))
                current_lines.clear()

        for line in lines:
            m = MarkdownChunker._HEADER_RE.match(line)
            if m:
                flush()
                hashes = m.group(1)
                title = m.group(2).strip()
                depth = MarkdownChunker._HEADER_DEPTH[hashes]
                for hdr_key in list(active_headers.keys()):
                    if MarkdownChunker._LEVEL_MAP[hdr_key] >= depth:
                        del active_headers[hdr_key]
                active_headers[MarkdownChunker._HEADER_LEVELS[hashes]] = title
            else:
                current_lines.append(line)

        flush()
        return sections

    def chunk(self, file_path: str, doc_type: DocType) -> list[DocumentChunk]:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()

        sections = self._parse_sections(text)
        chunks = []
        chunk_index = 0
        for headers, content in sections:
            if len(content) <= self._splitter._chunk_size:
                metadata = ChunkMetadata(
                    source=file_path,
                    filename=os.path.basename(file_path),
                    doc_type=doc_type,
                    format=FileFormat.MARKDOWN,
                    chunk_index=chunk_index,
                    headers=headers or None,
                )
                chunks.append(DocumentChunk(
                    chunk_id=DocumentChunk.generate_id(file_path, chunk_index),
                    page_content=content,
                    metadata=metadata,
                ))
                chunk_index += 1
            else:
                sub_splits = self._splitter.split_text(content)
                for sub in sub_splits:
                    metadata = ChunkMetadata(
                        source=file_path,
                        filename=os.path.basename(file_path),
                        doc_type=doc_type,
                        format=FileFormat.MARKDOWN,
                        chunk_index=chunk_index,
                        headers=headers or None,
                    )
                    chunks.append(DocumentChunk(
                        chunk_id=DocumentChunk.generate_id(file_path, chunk_index),
                        page_content=sub,
                        metadata=metadata,
                    ))
                    chunk_index += 1
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
