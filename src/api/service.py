import logging
import os
from dataclasses import asdict

from src.models import DocumentChunk
from src.retrieval.router import route_query
from src.retrieval.vector_store import HybridVectorStore

logger = logging.getLogger(__name__)


class IndexNotLoadedError(RuntimeError):
    pass


def _serialize_chunk(chunk: DocumentChunk) -> dict:
    d = asdict(chunk)
    d["metadata"]["doc_type"] = d["metadata"]["doc_type"].value
    d["metadata"]["format"] = d["metadata"]["format"].value
    return d


class RAGSearchService:
    def __init__(self, config: dict):
        self.config = config
        self.top_k = config["retrieval"]["top_k"]
        self._store: HybridVectorStore | None = None
        self._index_loaded = False

    @property
    def store(self) -> HybridVectorStore:
        if not self._index_loaded:
            self.ensure_loaded()
        return self._store

    @property
    def index_loaded(self) -> bool:
        return self._index_loaded

    def ensure_loaded(self) -> None:
        if self._index_loaded:
            return
        emb_cfg = self.config["embedding"]
        vector_db_dir = self.config["paths"]["vector_db_dir"]

        if not self.vector_index_exists(self.config):
            raise IndexNotLoadedError(
                f"No vector store found at {vector_db_dir}. Run 'ingest' first."
            )

        store = HybridVectorStore(
            embedding_model_name=emb_cfg["model"],
            device=emb_cfg.get("device", "cpu"),
        )
        if not store.load(vector_db_dir):
            logger.error(
                "No vector store found at %s. Run 'ingest' first.", vector_db_dir
            )
            raise IndexNotLoadedError(
                f"No vector store found at {vector_db_dir}. Run 'ingest' first."
            )
        self._store = store
        self._index_loaded = True
        logger.info("Vector store loaded (%d chunks)", len(store.chunks))

    @staticmethod
    def vector_index_exists(config: dict) -> bool:
        db_dir = config["paths"]["vector_db_dir"]
        return os.path.isfile(os.path.join(db_dir, "chunks.pkl"))

    def resolve_filter(self, query: str, doc_type: str) -> dict:
        if doc_type == "auto" or doc_type is None:
            return route_query(query)
        return {"doc_type": [doc_type]}

    def search(
        self, query: str, doc_type: str = "auto", top_k: int | None = None
    ) -> tuple[list[DocumentChunk], str, list[str]]:
        store = self.store
        k = top_k if top_k is not None else self.top_k
        filter_metadata = self.resolve_filter(query, doc_type)
        routed = filter_metadata.get("doc_type")

        results = store.hybrid_search(
            query=query,
            top_k=k,
            filter_metadata=filter_metadata,
        )

        used = doc_type if doc_type != "auto" else (
            routed[0] if isinstance(routed, list) and routed else "all"
        )
        routed_list = routed if isinstance(routed, list) else (
            [routed] if routed else None
        )
        return results, used, routed_list