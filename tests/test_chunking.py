import os
import tempfile

from src.ingestion.chunkers import (
    MarkdownChunker,
    PHPChunker,
)
from src.models import DocType


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
