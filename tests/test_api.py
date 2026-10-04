import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.config import load_config
from src.api.service import IndexNotLoadedError, RAGSearchService
from src.models import ChunkMetadata, DocType, DocumentChunk, FileFormat


def _make_empty_config(tmp_path) -> dict:
    return {
        "embedding": {"model": "BAAI/bge-m3", "device": "cpu"},
        "llm": {},
        "retrieval": {"top_k": 5},
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "vector_db_dir": str(tmp_path / "vector_db"),
        },
    }


def _fake_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="abc123",
        page_content="Função para resetar a senha no banco.",
        metadata=ChunkMetadata(
            source="data/raw/tech/Auth.php",
            filename="Auth.php",
            doc_type=DocType.TECH,
            format=FileFormat.PHP_CODE,
            chunk_index=0,
            detected_functions=["reset_password"],
        ),
    )


def _mock_store_class(monkeypatch):
    def fake_init(self, embedding_model_name="BAAI/bge-m3", device="cpu"):
        self.chunks = []
        self.embedder = None

    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.__init__", fake_init
    )
    monkeypatch.setattr(
        "src.api.service.RAGSearchService.vector_index_exists",
        staticmethod(lambda config: True),
    )


def _markdown_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="md001",
        page_content="## Instalação\n\nRode o comando de setup.",
        metadata=ChunkMetadata(
            source="data/raw/user/manual.md",
            filename="manual.md",
            doc_type=DocType.USER,
            format=FileFormat.MARKDOWN,
            chunk_index=2,
            headers={"h1": "Manual", "h2": "Instalação"},
        ),
    )


def _touch_index(config: dict) -> None:
    """Create the on-disk marker that vector_index_exists() looks for."""
    db_dir = Path(config["paths"]["vector_db_dir"])
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / "chunks.pkl").write_bytes(b"stub")


def _mock_loaded_store(monkeypatch, chunks) -> None:
    """Mock a loadable store, leaving vector_index_exists() on the real fs check."""
    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.__init__",
        lambda self, embedding_model_name="BAAI/bge-m3", device="cpu": setattr(
            self, "chunks", []
        ),
    )
    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.load",
        lambda self, db_dir: (setattr(self, "chunks", list(chunks)), True)[1],
    )


def test_health_without_index(tmp_path):
    config = _make_empty_config(tmp_path)
    app = create_app(config)
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["vector_index_loaded"] is False
    assert body["num_chunks"] == 0


def test_search_returns_503_when_no_index(tmp_path, monkeypatch):
    _mock_store_class(monkeypatch)
    config = _make_empty_config(tmp_path)
    app = create_app(config)
    client = TestClient(app)

    resp = client.post("/api/search", json={"query": "como resetar senha"})
    assert resp.status_code == 503


def test_search_doc_type_auto_routes_and_serializes(
    tmp_path, monkeypatch
):
    _mock_store_class(monkeypatch)
    fake = _fake_chunk()

    def fake_load(self, db_dir):
        self.chunks = [fake]
        return True

    def fake_hybrid_search(self, query, top_k=5, filter_metadata=None, alpha=0.5):
        return [fake]

    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.load", fake_load
    )
    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.hybrid_search",
        fake_hybrid_search,
    )

    config = _make_empty_config(tmp_path)
    app = create_app(config)
    client = TestClient(app)

    resp = client.post(
        "/api/search", json={"query": "bug no php", "doc_type": "auto"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["doc_type_used"] == "tech"
    assert body["routed_to"] == ["tech", "support"]

    result = body["results"][0]
    assert result["chunk_id"] == "abc123"
    assert result["metadata"]["doc_type"] == "tech"
    assert result["metadata"]["format"] == "php_code"
    assert result["metadata"]["detected_functions"] == ["reset_password"]


def test_search_explicit_doc_type(tmp_path, monkeypatch):
    _mock_store_class(monkeypatch)
    fake = _fake_chunk()

    def fake_load(self, db_dir):
        self.chunks = [fake]
        return True

    captured = {}

    def fake_hybrid_search(self, query, top_k=5, filter_metadata=None, alpha=0.5):
        captured["filter"] = filter_metadata
        return [fake]

    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.load", fake_load
    )
    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.hybrid_search",
        fake_hybrid_search,
    )

    config = _make_empty_config(tmp_path)
    app = create_app(config)
    client = TestClient(app)

    resp = client.post(
        "/api/search", json={"query": "alguma coisa", "doc_type": "user"}
    )
    assert resp.status_code == 200
    assert captured["filter"] == {"doc_type": ["user"]}
    assert resp.json()["doc_type_used"] == "user"
    assert resp.json()["routed_to"] == ["user"]


def test_search_invalid_query(tmp_path, monkeypatch):
    config = _make_empty_config(tmp_path)
    app = create_app(config)
    client = TestClient(app)

    resp = client.post("/api/search", json={"query": ""})
    assert resp.status_code == 422


def test_search_invalid_doc_type_returns_422(tmp_path):
    client = TestClient(create_app(_make_empty_config(tmp_path)))

    resp = client.post("/api/search", json={"query": "oi", "doc_type": "admin"})
    assert resp.status_code == 422


def test_health_reports_loaded_index(tmp_path, monkeypatch):
    _mock_loaded_store(monkeypatch, [_fake_chunk(), _markdown_chunk()])
    config = _make_empty_config(tmp_path)
    _touch_index(config)
    client = TestClient(create_app(config))

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "vector_index_loaded": True,
        "num_chunks": 2,
    }


def test_vector_index_exists_checks_chunks_pkl_on_disk(tmp_path):
    config = _make_empty_config(tmp_path)

    assert RAGSearchService.vector_index_exists(config) is False

    _touch_index(config)
    assert RAGSearchService.vector_index_exists(config) is True


def test_health_reports_not_loaded_when_store_fails_to_load(
    tmp_path, monkeypatch
):
    _mock_store_class(monkeypatch)
    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.load",
        lambda self, db_dir: False,
    )
    config = _make_empty_config(tmp_path)
    _touch_index(config)  # index file present, but load() fails
    client = TestClient(create_app(config))

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "vector_index_loaded": False,
        "num_chunks": 0,
    }


def test_search_top_k_defaults_to_config_and_accepts_override(
    tmp_path, monkeypatch
):
    seen_k = []
    _mock_loaded_store(monkeypatch, [_fake_chunk()])
    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.hybrid_search",
        lambda self, query, top_k=5, filter_metadata=None, alpha=0.5: (
            seen_k.append(top_k) or [_fake_chunk()]
        ),
    )
    config = _make_empty_config(tmp_path)  # retrieval.top_k == 5
    _touch_index(config)
    client = TestClient(create_app(config))

    assert client.post(
        "/api/search", json={"query": "resetar senha", "doc_type": "user"}
    ).status_code == 200
    assert client.post(
        "/api/search",
        json={"query": "resetar senha", "doc_type": "user", "top_k": 2},
    ).status_code == 200

    assert seen_k == [5, 2]


def test_search_without_route_matches_searches_all_doc_types(
    tmp_path, monkeypatch
):
    captured = {}
    _mock_loaded_store(monkeypatch, [_fake_chunk()])

    def fake_hybrid_search(self, query, top_k=5, filter_metadata=None, alpha=0.5):
        captured["filter"] = filter_metadata
        return []

    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.hybrid_search",
        fake_hybrid_search,
    )
    config = _make_empty_config(tmp_path)
    _touch_index(config)
    client = TestClient(create_app(config))

    resp = client.post("/api/search", json={"query": "zzz qqq"})
    assert resp.status_code == 200
    body = resp.json()
    assert captured["filter"] == {}
    assert body["count"] == 0
    assert body["results"] == []
    assert body["doc_type_used"] == "all"
    assert body["routed_to"] is None


def test_search_serializes_markdown_metadata(tmp_path, monkeypatch):
    md = _markdown_chunk()
    _mock_loaded_store(monkeypatch, [md])
    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.hybrid_search",
        lambda self, query, top_k=5, filter_metadata=None, alpha=0.5: [md],
    )
    config = _make_empty_config(tmp_path)
    _touch_index(config)
    client = TestClient(create_app(config))

    resp = client.post(
        "/api/search", json={"query": "como instalar", "doc_type": "user"}
    )
    assert resp.status_code == 200
    meta = resp.json()["results"][0]["metadata"]
    assert meta["doc_type"] == "user"
    assert meta["format"] == "markdown"
    assert meta["headers"] == {"h1": "Manual", "h2": "Instalação"}
    assert meta["page_number"] is None
    assert meta["detected_classes"] is None
    assert meta["detected_functions"] is None


def test_vector_store_is_loaded_once_per_process(tmp_path, monkeypatch):
    loads = []
    _mock_loaded_store(monkeypatch, [_fake_chunk()])
    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.load",
        lambda self, db_dir: (loads.append(db_dir), True)[1],
    )
    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.hybrid_search",
        lambda self, query, top_k=5, filter_metadata=None, alpha=0.5: [
            _fake_chunk()
        ],
    )
    config = _make_empty_config(tmp_path)
    _touch_index(config)
    service = RAGSearchService(config)

    assert service.index_loaded is False
    service.search("como usar o login", doc_type="user")
    service.search("como usar o login", doc_type="user")

    assert len(loads) == 1
    assert service.index_loaded is True


def test_search_raises_when_store_fails_to_load(tmp_path, monkeypatch):
    _mock_store_class(monkeypatch)
    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.load",
        lambda self, db_dir: False,
    )
    service = RAGSearchService(_make_empty_config(tmp_path))

    with pytest.raises(IndexNotLoadedError):
        service.search("qualquer coisa", doc_type="tech")


def test_api_load_config_reads_server_section(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "embedding:\n  model: BAAI/bge-m3\nserver:\n  host: 0.0.0.0\n  port: 9001\n",
        encoding="utf-8",
    )
    load_config.cache_clear()
    try:
        cfg = load_config(str(cfg_path))
    finally:
        load_config.cache_clear()

    assert cfg["server"] == {"host": "0.0.0.0", "port": 9001}


def _run_serve_main(monkeypatch, argv: list[str], config: dict) -> dict:
    import main as main_module

    calls: dict = {}
    monkeypatch.setattr(
        main_module, "load_config", lambda path="config.yaml": config
    )

    def fake_cmd_serve(cfg, host, port):
        calls.update(config=cfg, host=host, port=port)

    monkeypatch.setattr(main_module, "cmd_serve", fake_cmd_serve)
    monkeypatch.setattr(sys, "argv", argv)
    main_module.main()
    return calls


def test_serve_uses_host_and_port_from_config(tmp_path, monkeypatch):
    config = _make_empty_config(tmp_path)
    config["server"] = {"host": "127.0.0.1", "port": 8000}

    calls = _run_serve_main(monkeypatch, ["main.py", "serve"], config)

    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8000
    assert calls["config"] is config


def test_serve_cli_flags_override_config(tmp_path, monkeypatch):
    config = _make_empty_config(tmp_path)
    config["server"] = {"host": "127.0.0.1", "port": 8000}

    calls = _run_serve_main(
        monkeypatch,
        ["main.py", "serve", "--host", "0.0.0.0", "--port", "9999"],
        config,
    )

    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 9999


def test_serve_falls_back_to_defaults_without_server_block(
    tmp_path, monkeypatch
):
    config = _make_empty_config(tmp_path)  # no "server" key

    calls = _run_serve_main(monkeypatch, ["main.py", "serve"], config)

    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8000


def _poison_llm(monkeypatch) -> None:
    """Make any LLM/Ollama egress fail loudly instead of silently succeeding."""
    import requests

    from src.generation.llm_client import LLMClient

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "the REST API must stay retrieval-only: attempted an LLM call"
        )

    monkeypatch.setattr(requests, "post", forbidden)
    monkeypatch.setattr(requests, "get", forbidden)
    monkeypatch.setattr(LLMClient, "generate", forbidden)
    monkeypatch.setattr(LLMClient, "generate_stream", forbidden)


def test_api_endpoints_never_call_the_llm(tmp_path, monkeypatch):
    """Regression guard: /health and /api/search must not reach the LLM."""
    _poison_llm(monkeypatch)

    fake = _fake_chunk()
    _mock_loaded_store(monkeypatch, [fake])
    monkeypatch.setattr(
        "src.retrieval.vector_store.HybridVectorStore.hybrid_search",
        lambda self, query, top_k=5, filter_metadata=None, alpha=0.5: [fake],
    )
    config = _make_empty_config(tmp_path)
    _touch_index(config)
    client = TestClient(create_app(config))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["num_chunks"] == 1

    for doc_type in ("auto", "user", "tech", "support"):
        resp = client.post(
            "/api/search", json={"query": "resetar senha", "doc_type": doc_type}
        )
        assert resp.status_code == 200, doc_type
        assert resp.json()["count"] == 1


def test_api_error_paths_never_call_the_llm(tmp_path, monkeypatch):
    """The 503 and 422 paths must not fall back to the LLM either."""
    _poison_llm(monkeypatch)
    _mock_store_class(monkeypatch)
    client = TestClient(create_app(_make_empty_config(tmp_path)))

    assert client.get("/health").status_code == 200
    assert (
        client.post("/api/search", json={"query": "oi"}).status_code == 503
    )
    assert (
        client.post("/api/search", json={"query": ""}).status_code == 422
    )


def test_api_and_serve_do_not_import_the_llm_client():
    """Structural guard: the API process must not even load the LLM module."""
    code = (
        "import sys\n"
        "import src.api.app, src.api.service, src.api.schemas\n"
        "import main\n"
        "assert 'src.generation.llm_client' not in sys.modules, sorted(sys.modules)\n"
        "assert 'src.cli.repl' not in sys.modules, sorted(sys.modules)\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
