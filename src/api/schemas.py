from typing import Literal, Optional

from pydantic import BaseModel, Field

DocTypeFilter = Literal["auto", "user", "tech", "support"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    doc_type: DocTypeFilter = "auto"
    top_k: Optional[int] = None


class ChunkMetadataOut(BaseModel):
    source: str
    filename: str
    doc_type: str
    format: str
    chunk_index: int
    page_number: Optional[int] = None
    headers: Optional[dict[str, str]] = None
    detected_classes: Optional[list[str]] = None
    detected_functions: Optional[list[str]] = None


class SearchResult(BaseModel):
    chunk_id: str
    page_content: str
    metadata: ChunkMetadataOut


class SearchResponse(BaseModel):
    query: str
    doc_type_used: str
    routed_to: Optional[list[str]] = None
    count: int
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    vector_index_loaded: bool
    num_chunks: int = 0