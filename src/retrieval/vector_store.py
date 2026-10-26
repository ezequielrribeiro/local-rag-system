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

    def remove_sources(self, sources: set[str]) -> int:
        """Remove os chunks cujo `source` está no conjunto (CHG-R04)."""
        if not sources or not self.chunks:
            return 0

        keep = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.metadata.source not in sources
        ]
        removed = len(self.chunks) - len(keep)
        if removed == 0:
            return 0

        self.chunks = [self.chunks[index] for index in keep]
        if self.embeddings is not None:
            self.embeddings = self.embeddings[keep]
        self._rebuild_bm25()

        logger.info("Removed %d chunks from %d source(s)", removed, len(sources))
        return removed

    def _rebuild_bm25(self) -> None:
        self._tokenized_corpus = [
            chunk.page_content.lower().split() for chunk in self.chunks
        ]
        self.bm25 = (
            BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
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

        n = len(self.chunks)

        # O filtro de metadados é aplicado ANTES do corte de top_k, para que a
        # busca devolva até top_k resultados elegíveis (CHG-R01).
        eligible: list[int] | None = None
        if filter_metadata:
            eligible = []
            for idx, chunk in enumerate(self.chunks):
                match = all(
                    getattr(chunk.metadata, key, None) in value
                    if isinstance(value, list)
                    else getattr(chunk.metadata, key, None) == value
                    for key, value in filter_metadata.items()
                )
                if match:
                    eligible.append(idx)

            if not eligible:
                return []

        dense_ranks = [int(i) for i in np.argsort(-dense_scores)]
        bm25_ranks = [int(i) for i in np.argsort(-bm25_scores)]

        if eligible is not None:
            eligible_set = set(eligible)
            dense_ranks = [i for i in dense_ranks if i in eligible_set]
            bm25_ranks = [i for i in bm25_ranks if i in eligible_set]

        rrf_scores = np.zeros(n)
        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] += alpha / (60 + rank + 1)
        for rank, idx in enumerate(bm25_ranks):
            rrf_scores[idx] += (1 - alpha) / (60 + rank + 1)

        candidates = eligible if eligible is not None else list(range(n))
        top_indices = sorted(candidates, key=lambda i: -rrf_scores[i])[:top_k]

        return [self.chunks[i] for i in top_indices]

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
