import ast
import logging
import os
import re
from abc import ABC, abstractmethod

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.models import ChunkMetadata, DocType, DocumentChunk, FileFormat

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    VENDOR_EXCLUDE_PATTERNS = [
        re.compile(r"[/\\]vendor[/\\]"),
        re.compile(r"[/\\]node_modules[/\\]"),
        re.compile(r"[/\\]\.venv[/\\]"),
        re.compile(r"[/\\]venv[/\\]"),
        re.compile(r"[/\\]__pycache__[/\\]"),
        re.compile(r"[/\\]site-packages[/\\]"),
        re.compile(r"[/\\]dist[/\\]"),
        re.compile(r"[/\\]build[/\\]"),
        re.compile(r"[/\\]\.next[/\\]"),
        re.compile(r"[/\\]coverage[/\\]"),
    ]
    VENDOR_EXCLUDE_FILENAME_RE = re.compile(
        r"\.(?:min|bundle|pack)\.js$", re.IGNORECASE
    )

    def should_exclude(self, file_path: str) -> bool:
        for pattern in self.VENDOR_EXCLUDE_PATTERNS:
            if pattern.search(file_path):
                return True
        basename = os.path.basename(file_path.replace("\\", "/"))
        return bool(self.VENDOR_EXCLUDE_FILENAME_RE.search(basename))

    @abstractmethod
    def chunk(self, file_path: str, doc_type: DocType) -> list[DocumentChunk]:
        ...


class CodeChunker(BaseChunker):
    """Shared language-aware chunking for source files.

    Subclasses declare FORMAT, SEPARATORS and symbol detection; this class
    handles splitting oversized code and attaching the DocumentChunk metadata.
    """

    FORMAT: FileFormat = FileFormat.PHP_CODE
    SEPARATORS: list[str] = ["\n\n", "\n", "."]
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._code_splitter = RecursiveCharacterTextSplitter(
            separators=list(self.SEPARATORS),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _detect_classes(self, text: str) -> list[str]:
        raise NotImplementedError

    def _detect_functions(self, text: str) -> list[str]:
        raise NotImplementedError

    def _make_chunk(
        self,
        text: str,
        file_path: str,
        filename: str,
        doc_type: DocType,
        chunk_index: int,
        fallback_classes: list[str] | None,
        fallback_functions: list[str] | None,
    ) -> DocumentChunk:
        chunk_classes = self._detect_classes(text)
        chunk_functions = self._detect_functions(text)
        metadata = ChunkMetadata(
            source=file_path,
            filename=filename,
            doc_type=doc_type,
            format=self.FORMAT,
            chunk_index=chunk_index,
            detected_classes=chunk_classes or fallback_classes,
            detected_functions=chunk_functions or fallback_functions,
        )
        return DocumentChunk(
            chunk_id=DocumentChunk.generate_id(file_path, chunk_index),
            page_content=text,
            metadata=metadata,
        )

    def _build_chunks(
        self,
        code: str,
        file_path: str,
        filename: str,
        doc_type: DocType,
        chunk_index: int,
        full_classes: list[str] | None = None,
        full_functions: list[str] | None = None,
    ) -> tuple[list[DocumentChunk], int]:
        """Split a whole code blob, tagging symbols found in each piece."""
        if full_classes is None:
            full_classes = self._detect_classes(code)
        if full_functions is None:
            full_functions = self._detect_functions(code)

        chunks: list[DocumentChunk] = []
        for i, lc_doc in enumerate(self._code_splitter.create_documents([code])):
            chunks.append(
                self._make_chunk(
                    lc_doc.page_content,
                    file_path,
                    filename,
                    doc_type,
                    chunk_index,
                    full_classes if i == 0 else None,
                    full_functions if i == 0 else None,
                )
            )
            chunk_index += 1
        return chunks, chunk_index

    def _build_chunks_from_units(
        self,
        units: list[str],
        file_path: str,
        filename: str,
        doc_type: DocType,
        chunk_index: int,
        full_classes: list[str],
        full_functions: list[str],
    ) -> tuple[list[DocumentChunk], int]:
        """Emit one chunk per structural unit, sub-splitting oversized ones."""
        chunks: list[DocumentChunk] = []
        for unit in units:
            if len(unit) <= self.chunk_size:
                pieces = [unit]
            else:
                pieces = self._code_splitter.split_text(unit)
            for j, piece in enumerate(pieces):
                if not piece.strip():
                    continue
                is_first = not chunks and j == 0
                chunks.append(
                    self._make_chunk(
                        piece,
                        file_path,
                        filename,
                        doc_type,
                        chunk_index,
                        full_classes if is_first else None,
                        full_functions if is_first else None,
                    )
                )
                chunk_index += 1
        return chunks, chunk_index


class PHPChunker(CodeChunker):
    _CLASS_REGEX = re.compile(r"(?:^|\n)\s*(?:abstract\s+)?(?:final\s+)?class\s+(\w+)")
    _FUNCTION_REGEX = re.compile(
        r"(?:^|\n)\s*(?:public|protected|private|static|\s)*\s*function\s+(\w+)\s*\("
    )
    _PHP_OPEN_RE = re.compile(r"<\?(?:php)?\s*")
    _PHP_CLOSE_RE = re.compile(r"\?>")
    FORMAT = FileFormat.PHP_CODE
    SEPARATORS = [
        "\nclass ",
        "\nfunction ",
        "\npublic function ",
        "\nprotected function ",
        "\nprivate function ",
        "\n\n",
        "\n",
        ".",
    ]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        super().__init__(chunk_size, chunk_overlap)
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
        return self._build_chunks(
            php_code, file_path, filename, doc_type, chunk_index
        )

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


class PythonChunker(CodeChunker):
    """AST-driven chunking for Python sources.

    Uses the stdlib ``ast`` module for exact def/class boundaries and symbol
    detection, falling back to regex detection when a file does not parse.
    """

    FORMAT = FileFormat.PYTHON_CODE
    SEPARATORS = [
        "\ndef ",
        "\nasync def ",
        "\nclass ",
        "\n    def ",
        "\n    async def ",
        "\n\n",
        "\n",
        ".",
    ]
    _CLASS_REGEX = re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)
    _FUNCTION_REGEX = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.MULTILINE)
    _DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def _parse(self, source: str) -> ast.Module | None:
        try:
            return ast.parse(source)
        except (SyntaxError, ValueError):
            return None

    def _detect_symbols(self, tree: ast.AST) -> tuple[list[str], list[str]]:
        classes: list[str] = []
        functions: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
        return classes, functions

    def _detect_classes(self, text: str) -> list[str]:
        tree = self._parse(text)
        if tree is not None:
            classes, _functions = self._detect_symbols(tree)
            return classes
        return self._CLASS_REGEX.findall(text)

    def _detect_functions(self, text: str) -> list[str]:
        tree = self._parse(text)
        if tree is not None:
            _classes, functions = self._detect_symbols(tree)
            return functions
        return self._FUNCTION_REGEX.findall(text)

    @staticmethod
    def _unit_start(node: ast.stmt, lines: list[str]) -> int:
        """0-based first line of a statement, including decorators/comments."""
        start = node.lineno - 1
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min([start, *(d.lineno - 1 for d in decorators)])
        while start > 0 and lines[start - 1].strip().startswith("#"):
            start -= 1
        return start

    def _code_units(self, source: str, tree: ast.Module) -> list[str]:
        """Split a module into def/class units plus a leading preamble unit."""
        lines = source.splitlines(keepends=True)
        units: list[str] = []
        pending: list[str] = []
        consumed = 0

        def flush() -> None:
            text = "".join(pending).strip()
            if text:
                units.append(text)
            pending.clear()

        for node in tree.body:
            start = self._unit_start(node, lines)
            end = getattr(node, "end_lineno", None) or node.lineno
            if start > consumed:
                pending.extend(lines[consumed:start])
            segment = "".join(lines[start:end])
            if isinstance(node, self._DEFINITION_NODES):
                flush()
                units.append(segment.strip())
            else:
                pending.append(segment)
            consumed = max(consumed, end)

        pending.extend(lines[consumed:])
        flush()
        return [unit for unit in units if unit]

    def chunk(self, file_path: str, doc_type: DocType) -> list[DocumentChunk]:
        if self.should_exclude(file_path):
            return []

        with open(file_path, encoding="utf-8", errors="replace") as f:
            source = f.read()

        filename = os.path.basename(file_path)
        tree = self._parse(source)
        if tree is None:
            logger.warning(
                "Python source does not parse, falling back to text splitting: %s",
                file_path,
            )
            chunks, _ = self._build_chunks(
                source, file_path, filename, doc_type, 0
            )
            return chunks

        classes, functions = self._detect_symbols(tree)
        units = self._code_units(source, tree)
        if not units:
            return []

        chunks, _ = self._build_chunks_from_units(
            units,
            file_path,
            filename,
            doc_type,
            0,
            classes,
            functions,
        )
        return chunks


class JavaScriptChunker(CodeChunker):
    """Regex-driven chunking for JavaScript sources.

    There is no stdlib parser for JS, so boundaries come from the recursive
    splitter using declaration-aware separators and symbols from regexes.
    """

    FORMAT = FileFormat.JAVASCRIPT_CODE
    SEPARATORS = [
        "\nexport default class ",
        "\nexport class ",
        "\nclass ",
        "\nexport async function ",
        "\nexport function ",
        "\nasync function ",
        "\nfunction ",
        "\nexport const ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        "\n\n",
        "\n",
        ".",
    ]
    _CLASS_REGEX = re.compile(
        r"(?:^|\n)\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?"
        r"class\s+([A-Za-z_$][\w$]*)"
    )
    _DECLARED_FUNCTION_RE = re.compile(
        r"(?:^|\n)\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"function\s*\*?\s*([A-Za-z_$][\w$]*)"
    )
    _ASSIGNED_FUNCTION_RE = re.compile(
        r"(?:^|\n)\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
        r"(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
    )
    _METHOD_RE = re.compile(
        r"(?:^|\n)\s{2,}(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?\*?\s*"
        r"([A-Za-z_$][\w$]*)\s*\([^;{)]*\)\s*\{"
    )
    _RESERVED_METHOD_NAMES = {
        "if", "for", "while", "switch", "catch", "function", "return", "do",
        "else", "try", "with",
    }

    def _detect_classes(self, text: str) -> list[str]:
        return self._CLASS_REGEX.findall(text)

    def _detect_functions(self, text: str) -> list[str]:
        found: list[tuple[int, str]] = []
        for match in self._DECLARED_FUNCTION_RE.finditer(text):
            found.append((match.start(), match.group(1)))
        for match in self._ASSIGNED_FUNCTION_RE.finditer(text):
            found.append((match.start(), match.group(1)))
        for match in self._METHOD_RE.finditer(text):
            name = match.group(1)
            if name in self._RESERVED_METHOD_NAMES:
                continue
            found.append((match.start(), name))
        found.sort(key=lambda item: item[0])
        return list(dict.fromkeys(name for _pos, name in found))

    def chunk(self, file_path: str, doc_type: DocType) -> list[DocumentChunk]:
        if self.should_exclude(file_path):
            return []

        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()

        filename = os.path.basename(file_path)
        chunks, _ = self._build_chunks(
            text, file_path, filename, doc_type, 0
        )
        return chunks


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
