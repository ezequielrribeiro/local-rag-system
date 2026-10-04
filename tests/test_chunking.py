import os
import re
import tempfile

from src.ingestion.chunkers import (
    JavaScriptChunker,
    MarkdownChunker,
    PHPChunker,
    PythonChunker,
)
from src.models import DocType


def _write_temp(content: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _assert_no_source_loss(source: str, chunks) -> None:
    """Every meaningful source line must survive in some chunk."""
    rebuilt = re.sub(r"\s+", " ", "".join(c.page_content for c in chunks))
    for line in source.splitlines():
        stripped = line.strip()
        if len(stripped) < 8:
            continue
        assert re.sub(r"\s+", " ", stripped) in rebuilt, (
            f"source line dropped by chunking: {stripped}"
        )


def test_php_vendor_exclusion():
    chunker = PHPChunker()
    assert chunker.should_exclude("data/raw/tech/vendor/guzzlehttp/GuzzleClient.php")
    assert chunker.should_exclude("data/raw/tech/node_modules/package/index.php")
    assert not chunker.should_exclude("data/raw/tech/controllers/UserController.php")


def test_php_vendor_exclusion_using_pipeline():
    from src.ingestion.pipeline import run_ingestion

    with tempfile.TemporaryDirectory() as tmp:
        tech_dir = os.path.join(tmp, "tech")
        vendor_dir = os.path.join(tech_dir, "vendor", "guzzlehttp")
        os.makedirs(vendor_dir)
        os.makedirs(os.path.join(tech_dir, "controllers"))

        vendor_file = os.path.join(vendor_dir, "GuzzleClient.php")
        with open(vendor_file, "w") as f:
            f.write("<?php\nclass GuzzleClient {}\n")

        controller_file = os.path.join(tech_dir, "controllers", "UserController.php")
        with open(controller_file, "w") as f:
            f.write(
                "<?php\nclass UserController {\n"
                "    public function login() {}\n"
                "    public function logout() {}\n"
                "}\n"
            )

        chunks = run_ingestion(raw_dir=tmp, processed_dir=tmp)
        assert all("/vendor/" not in c.metadata.source for c in chunks)
        assert all(
            c.metadata.format.value == "php_code"
            for c in chunks
            if "UserController" in c.metadata.filename
        )


def test_php_detects_classes_and_functions():
    chunker = PHPChunker()
    doc_type = DocType.TECH

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".php", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "<?php\n"
            "class AuthService {\n"
            "    public function authenticate($user) {}\n"
            "    private function validateToken($token) {}\n"
            "    protected function hashPassword($pass) {}\n"
            "}\n"
        )
        tmppath = f.name

    try:
        chunks = chunker.chunk(tmppath, doc_type)
        assert len(chunks) > 0
        first = chunks[0]
        assert first.metadata.detected_classes == ["AuthService"]
        assert "authenticate" in (first.metadata.detected_functions or [])
    finally:
        os.unlink(tmppath)


def test_markdown_heading_splitting():
    chunker = MarkdownChunker()
    doc_type = DocType.USER

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "# Welcome\n\nIntroduction text.\n\n"
            "## Installation\n\nInstall steps.\n\n"
            "### Prerequisites\n\nPrereq details.\n\n"
            "## Usage\n\nUsage instructions.\n"
        )
        tmppath = f.name

    try:
        chunks = chunker.chunk(tmppath, doc_type)
        assert len(chunks) >= 2
        headers_section1 = chunks[0].metadata.headers
        assert headers_section1 is None or "Header 1" in str(headers_section1)
    finally:
        os.unlink(tmppath)


PYTHON_SOURCE = '''import os

DEFAULT_TIMEOUT = 30


# helper kept with the function below
def retry(fn, attempts=3):
    for _ in range(attempts):
        try:
            return fn()
        except Exception:
            continue


class AuthService:
    def authenticate(self, user):
        return True

    def validate_token(self, token):
        return bool(token)
'''

JAVASCRIPT_SOURCE = """const util = require('util');

export default class AuthService {
  constructor(opts) {
    this.opts = opts;
  }

  async authenticate(user) {
    return true;
  }
}

export function hashPassword(pass) {
  return pass;
}

const validateToken = async (token) => {
  return Boolean(token);
};

function legacyHelper(a, b) {
  return a + b;
}
"""


def test_python_vendor_exclusion():
    chunker = PythonChunker()
    for path in [
        "data/raw/tech/.venv/lib/site.py",
        "data/raw/tech/venv/lib/site.py",
        "data/raw/tech/pkg/__pycache__/mod.cpython-313.py",
        "data/raw/tech/site-packages/requests/api.py",
        "data/raw/tech/node_modules/pkg/index.py",
        "data/raw/tech/vendor/lib/thing.py",
        "C:\\projetos\\data\\raw\\tech\\.venv\\lib\\site.py",
    ]:
        assert chunker.should_exclude(path), path

    for path in [
        "data/raw/tech/src/service.py",
        "data/raw/tech/src/myvenv/app.py",
        "data/raw/tech/src/venvish/app.py",
        "data/raw/tech/src/builder.py",
    ]:
        assert not chunker.should_exclude(path), path


def test_javascript_excludes_minified_and_bundled_files():
    chunker = JavaScriptChunker()
    for path in [
        "data/raw/tech/public/app.min.js",
        "data/raw/tech/dist/vendor.bundle.js",
        "data/raw/tech/dist/vendor.pack.js",
        "data/raw/tech/public/APP.MIN.JS",
        "data/raw/tech/node_modules/left-pad/index.js",
        "data/raw/tech/dist/build.js",
    ]:
        assert chunker.should_exclude(path), path

    for path in [
        "data/raw/tech/src/app.js",
        "data/raw/tech/src/minimal.js",
        "data/raw/tech/src/bundle.js",
    ]:
        assert not chunker.should_exclude(path), path


def test_python_detects_classes_and_functions():
    chunker = PythonChunker()
    tmppath = _write_temp(PYTHON_SOURCE, ".py")
    try:
        chunks = chunker.chunk(tmppath, DocType.TECH)
        assert len(chunks) > 0
        first = chunks[0]
        assert first.metadata.format.value == "python_code"
        assert first.metadata.detected_classes == ["AuthService"]
        assert first.metadata.detected_functions == [
            "retry",
            "authenticate",
            "validate_token",
        ]
    finally:
        os.unlink(tmppath)


def test_python_splits_into_definition_units():
    chunker = PythonChunker()
    tmppath = _write_temp(PYTHON_SOURCE, ".py")
    try:
        chunks = chunker.chunk(tmppath, DocType.TECH)
        contents = [c.page_content for c in chunks]

        assert any(c.startswith("import os") for c in contents)
        assert any("def retry(" in c for c in contents)
        assert any("class AuthService:" in c for c in contents)

        # module preamble stays separate from the function definition
        preamble = next(c for c in contents if c.startswith("import os"))
        assert "def retry" not in preamble

        # the comment above retry() travels with the function
        retry_chunk = next(c for c in contents if "def retry(" in c)
        assert "# helper kept with the function below" in retry_chunk
    finally:
        os.unlink(tmppath)


def test_python_large_class_is_sub_split():
    chunker = PythonChunker(chunk_size=300, chunk_overlap=32)
    source = "class Big:\n" + "".join(
        f"    def method_{i}(self):\n        return {i}\n\n" for i in range(30)
    )
    tmppath = _write_temp(source, ".py")
    try:
        chunks = chunker.chunk(tmppath, DocType.TECH)
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 320 for c in chunks)
        assert chunks[0].metadata.detected_classes == ["Big"]
        # only the first chunk carries the file-wide symbol fallback
        assert all(
            c.metadata.detected_classes is None for c in chunks[1:]
        )
    finally:
        os.unlink(tmppath)


def test_python_falls_back_when_source_does_not_parse():
    chunker = PythonChunker()
    tmppath = _write_temp("def broken(:\n    class ???\n", ".py")
    try:
        chunks = chunker.chunk(tmppath, DocType.TECH)
        assert len(chunks) > 0
        assert all(
            c.metadata.format.value == "python_code" for c in chunks
        )
    finally:
        os.unlink(tmppath)


def test_python_empty_file_yields_no_chunks():
    chunker = PythonChunker()
    tmppath = _write_temp("", ".py")
    try:
        assert chunker.chunk(tmppath, DocType.TECH) == []
    finally:
        os.unlink(tmppath)


def test_javascript_detects_classes_and_functions():
    chunker = JavaScriptChunker()
    tmppath = _write_temp(JAVASCRIPT_SOURCE, ".js")
    try:
        chunks = chunker.chunk(tmppath, DocType.TECH)
        assert len(chunks) > 0
        first = chunks[0]
        assert first.metadata.format.value == "javascript_code"
        assert first.metadata.detected_classes == ["AuthService"]

        functions = first.metadata.detected_functions or []
        assert "hashPassword" in functions  # exported declaration
        assert "legacyHelper" in functions  # plain declaration
        assert "validateToken" in functions  # arrow function
        assert "authenticate" in functions  # class method
        assert "constructor" in functions
    finally:
        os.unlink(tmppath)


def test_javascript_ignores_control_flow_as_methods():
    chunker = JavaScriptChunker()
    source = (
        "function run(items) {\n"
        "  for (const item of items) {\n"
        "    if (item) {\n"
        "      while (item) {\n"
        "        try { item(); } catch (e) {}\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    tmppath = _write_temp(source, ".js")
    try:
        chunks = chunker.chunk(tmppath, DocType.TECH)
        functions = chunks[0].metadata.detected_functions or []
        assert functions == ["run"]
    finally:
        os.unlink(tmppath)


def test_javascript_empty_file_yields_no_chunks():
    chunker = JavaScriptChunker()
    tmppath = _write_temp("", ".js")
    try:
        assert chunker.chunk(tmppath, DocType.TECH) == []
    finally:
        os.unlink(tmppath)


def test_python_keeps_decorators_with_their_definition():
    source = (
        "@dataclass\n"
        "class Config:\n"
        + "".join(f"    field_{i}: int = {i}\n" for i in range(20))
        + "\n    def total(self) -> int:\n        return 0\n"
    )
    tmppath = _write_temp(source, ".py")
    try:
        chunks = PythonChunker(chunk_size=200, chunk_overlap=32).chunk(
            tmppath, DocType.TECH
        )
        decorator_chunk = next(
            c for c in chunks if "@dataclass" in c.page_content
        )
        assert "class Config:" in decorator_chunk.page_content
        _assert_no_source_loss(source, chunks)
    finally:
        os.unlink(tmppath)


def test_python_keeps_signature_with_its_docstring():
    source = (
        "def process(items):\n"
        '    """Process a batch of items carefully."""\n'
        + "".join(f"    value_{i} = compute({i})\n" for i in range(30))
        + "    return items\n"
    )
    tmppath = _write_temp(source, ".py")
    try:
        chunks = PythonChunker(chunk_size=300, chunk_overlap=32).chunk(
            tmppath, DocType.TECH
        )
        signature_chunk = next(
            c for c in chunks if "def process(items):" in c.page_content
        )
        assert "Process a batch of items carefully." in signature_chunk.page_content
        _assert_no_source_loss(source, chunks)
    finally:
        os.unlink(tmppath)


def test_python_chunking_never_drops_source_lines():
    source = (
        PYTHON_SOURCE
        + "\n\nclass Report:\n"
        + "".join(
            f"    def column_{i}(self):\n        return {i}\n\n"
            for i in range(40)
        )
    )
    tmppath = _write_temp(source, ".py")
    try:
        chunks = PythonChunker().chunk(tmppath, DocType.TECH)
        assert len(chunks) > 1
        _assert_no_source_loss(source, chunks)
    finally:
        os.unlink(tmppath)


def test_javascript_chunking_never_drops_source_lines():
    tmppath = _write_temp(JAVASCRIPT_SOURCE, ".js")
    try:
        chunks = JavaScriptChunker(chunk_size=200, chunk_overlap=32).chunk(
            tmppath, DocType.TECH
        )
        assert len(chunks) > 1
        _assert_no_source_loss(JAVASCRIPT_SOURCE, chunks)
    finally:
        os.unlink(tmppath)


def test_pipeline_ingests_python_and_javascript():
    from src.ingestion.pipeline import run_ingestion

    with tempfile.TemporaryDirectory() as tmp:
        tech_dir = os.path.join(tmp, "tech")
        src_dir = os.path.join(tech_dir, "src")
        venv_dir = os.path.join(tech_dir, ".venv", "lib")
        node_dir = os.path.join(tech_dir, "node_modules", "pkg")
        os.makedirs(src_dir)
        os.makedirs(venv_dir)
        os.makedirs(node_dir)

        def write(path: str, content: str) -> None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        write(
            os.path.join(src_dir, "service.py"),
            "class Service:\n    def run(self):\n        return 1\n",
        )
        write(
            os.path.join(src_dir, "app.js"),
            "export function boot() {\n  return 1;\n}\n",
        )
        write(os.path.join(venv_dir, "vendored.py"), "class V: pass\n")
        write(os.path.join(node_dir, "dep.js"), "module.exports = 1;\n")
        write(os.path.join(src_dir, "app.min.js"), "var a=1;\n")

        chunks = run_ingestion(raw_dir=tmp, processed_dir=tmp)

        formats = {c.metadata.filename: c.metadata.format.value for c in chunks}
        assert formats == {
            "service.py": "python_code",
            "app.js": "javascript_code",
        }
        assert all(".venv" not in c.metadata.source for c in chunks)
        assert all("node_modules" not in c.metadata.source for c in chunks)
        assert all(".min.js" not in c.metadata.source for c in chunks)

        python_chunk = next(
            c for c in chunks if c.metadata.filename == "service.py"
        )
        assert python_chunk.metadata.detected_classes == ["Service"]
        assert python_chunk.metadata.detected_functions == ["run"]

        js_chunk = next(c for c in chunks if c.metadata.filename == "app.js")
        assert js_chunk.metadata.detected_functions == ["boot"]
