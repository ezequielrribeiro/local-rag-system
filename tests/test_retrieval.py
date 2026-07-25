from src.generation.llm_client import LLMClient, NO_CONTEXT_MESSAGE
from src.ingestion.chunkers import MarkdownChunker, PHPChunker
from src.models import DocType, DocumentChunk
from src.retrieval.router import route_query


def test_query_router_tech_domain():
    filter_meta = route_query("Como corrigir um bug no código PHP?")
    assert "doc_type" in filter_meta
    assert "tech" in filter_meta["doc_type"]
    assert "user" not in filter_meta["doc_type"]


def test_query_router_user_domain():
    filter_meta = route_query("Como usar o manual de login passo a passo?")
    assert "doc_type" in filter_meta
    assert "user" in filter_meta["doc_type"]
    assert "support" in filter_meta["doc_type"]


def test_query_router_mixed_domain():
    filter_meta = route_query("Qual a função para login?")
    assert "doc_type" in filter_meta


def test_hallucination_prevention_empty_context():
    client = LLMClient()
    result = client.generate(
        query="Como funciona o checkout com Pix no sistema?",
        context_chunks=[],
        doc_type="tech",
    )
    assert NO_CONTEXT_MESSAGE in result


def test_hallucination_prevention_none_context():
    client = LLMClient()
    result = client.generate(
        query="Como funciona o módulo de Pix?",
        context_chunks=[],
        doc_type="support",
    )
    assert NO_CONTEXT_MESSAGE in result
