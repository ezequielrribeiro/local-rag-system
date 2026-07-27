import gc
import json
import logging
import os
import pickle

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from src.models import DocumentChunk

logger = logging.getLogger(__name__)


class HybridVectorStore:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
    ):
        logger.info("Loading embedding model: %s", embedding_model_name)
        self.embedder = SentenceTransformer(embedding_model_name, device=device)
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return

        texts = [c.page_content for c in chunks]
        logger.info("Generating embeddings for %d chunks...", len(texts))
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        embeddings = normalize(embeddings, axis=1)

        if self.embeddings is None:
            self.embeddings = embeddings
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.chunks.extend(chunks)

        tokenized = [text.lower().split() for text in texts]
        if self.bm25 is None:
            self._tokenized_corpus = tokenized
            self.bm25 = BM25Okapi(tokenized)
        else:
            self._tokenized_corpus.extend(tokenized)
            self.bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(
            "Vector store now has %d documents", len(self.chunks)
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict | None = None,
        alpha: float = 0.5,
    ) -> list[DocumentChunk]:
        query_embedding = self.embedder.encode([query])
        query_embedding = normalize(query_embedding, axis=1)

        dense_scores = (self.embeddings @ query_embedding.T).flatten()

        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        dense_ranks = np.argsort(-dense_scores)
        bm25_ranks = np.argsort(-bm25_scores)

        n = len(self.chunks)
        rrf_scores = np.zeros(n)
        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] += alpha / (60 + rank + 1)
        for rank, idx in enumerate(bm25_ranks):
            rrf_scores[idx] += (1 - alpha) / (60 + rank + 1)

        top_indices = np.argsort(-rrf_scores)[:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            if filter_metadata:
                match = all(
                    getattr(chunk.metadata, k, None) == v
                    if not isinstance(v, list)
                    else getattr(chunk.metadata, k, None) in v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue
            results.append(chunk)

        return results

    def save(self, db_dir: str) -> None:
        os.makedirs(db_dir, exist_ok=True)

        with open(os.path.join(db_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        if self.embeddings is not None:
            np.save(os.path.join(db_dir, "embeddings.npy"), self.embeddings)

        with open(os.path.join(db_dir, "tokenized_corpus.pkl"), "wb") as f:
            pickle.dump(self._tokenized_corpus, f)

        metadata = {"num_chunks": len(self.chunks)}
        with open(os.path.join(db_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        logger.info("Vector store saved to %s (%d chunks)", db_dir, len(self.chunks))

    def load(self, db_dir: str) -> bool:
        chunks_path = os.path.join(db_dir, "chunks.pkl")
        embeddings_path = os.path.join(db_dir, "embeddings.npy")
        tokenized_path = os.path.join(db_dir, "tokenized_corpus.pkl")

        if not all(os.path.exists(p) for p in [chunks_path, embeddings_path, tokenized_path]):
            return False

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.embeddings = np.load(embeddings_path)

        with open(tokenized_path, "rb") as f:
            self._tokenized_corpus = pickle.load(f)

        self.bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(
            "Vector store loaded from %s (%d chunks)", db_dir, len(self.chunks)
        )
        return True

    def clear(self) -> None:
        logger.info("Clearing vector store from memory...")
        self.chunks.clear()
        self.embeddings = None
        if self.bm25 is not None:
            self.bm25 = None
        self._tokenized_corpus.clear()
        if self.embedder is not None:
            del self.embedder
            self.embedder = None
        gc.collect()
        logger.info("Vector store memory released.")
