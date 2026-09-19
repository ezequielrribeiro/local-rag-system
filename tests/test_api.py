from fastapi.testclient import TestClient

from src.api.app import create_app
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