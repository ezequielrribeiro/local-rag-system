import logging

from fastapi import Depends, FastAPI, HTTPException, Request

from src.api.config import load_config
from src.api.schemas import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.api.service import IndexNotLoadedError, RAGSearchService, _serialize_chunk

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def get_service(request: Request) -> RAGSearchService:
    return request.app.state.service


def create_app(config: dict | None = None) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI(title="Local RAG System API", version="1.0.0")
    app.state.service = RAGSearchService(cfg)
    app.state.config = cfg

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        service: RAGSearchService = app.state.service
        if not RAGSearchService.vector_index_exists(cfg):
            return HealthResponse(
                status="ok", vector_index_loaded=False, num_chunks=0
            )

        try:
            num_chunks = len(service.store.chunks)
        except IndexNotLoadedError:
            logger.warning("Vector index present on disk but could not be loaded.")
            return HealthResponse(
                status="ok", vector_index_loaded=False, num_chunks=0
            )

        return HealthResponse(
            status="ok", vector_index_loaded=True, num_chunks=num_chunks
        )

    @app.post("/api/search", response_model=SearchResponse)
    def search(
        payload: SearchRequest,
        service: RAGSearchService = Depends(get_service),
    ) -> SearchResponse:
        try:
            results, used, routed = service.search(
                query=payload.query,
                doc_type=payload.doc_type,
                top_k=payload.top_k,
            )
        except IndexNotLoadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        return SearchResponse(
            query=payload.query,
            doc_type_used=used,
            routed_to=routed,
            count=len(results),
            results=[SearchResult(**_serialize_chunk(c)) for c in results],
        )

    return app