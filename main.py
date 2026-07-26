import argparse
import logging
import sys

import yaml

from src.generation.llm_client import LLMClient

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("main")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_ingest(config: dict) -> None:
    from src.ingestion.pipeline import run_ingestion
    from src.retrieval.vector_store import HybridVectorStore

    raw_dir = config["paths"]["raw_dir"]
    processed_dir = config["paths"]["processed_dir"]
    vector_db_dir = config["paths"]["vector_db_dir"]
    emb_cfg = config["embedding"]

    chunks = run_ingestion(raw_dir=raw_dir, processed_dir=processed_dir)
    if not chunks:
        logger.warning("No chunks generated from ingestion.")
        return

    store = HybridVectorStore(
        embedding_model_name=emb_cfg["model"],
        device=emb_cfg.get("device", "cpu"),
    )
    store.add_documents(chunks)
    store.save(vector_db_dir)
    logger.info("Ingestion and indexing complete.")


def cmd_query(config: dict, query: str, doc_type_override: str | None) -> None:
    from src.retrieval.router import route_query
    from src.retrieval.vector_store import HybridVectorStore

    vector_db_dir = config["paths"]["vector_db_dir"]
    emb_cfg = config["embedding"]
    ret_cfg = config["retrieval"]
    llm_cfg = config["llm"]

    store = HybridVectorStore(
        embedding_model_name=emb_cfg["model"],
        device=emb_cfg.get("device", "cpu"),
    )
    loaded = store.load(vector_db_dir)
    if not loaded:
        logger.error(
            "No vector store found at %s. Run 'ingest' first.", vector_db_dir
        )
        sys.exit(1)

    if doc_type_override:
        filter_metadata = {"doc_type": doc_type_override}
    else:
        filter_metadata = route_query(query)

    results = store.hybrid_search(
        query=query,
        top_k=ret_cfg["top_k"],
        filter_metadata=filter_metadata,
    )

    doc_type = (
        doc_type_override
        or (
            filter_metadata.get("doc_type")
            and (
                filter_metadata["doc_type"][0]
                if isinstance(filter_metadata["doc_type"], list)
                else filter_metadata["doc_type"]
            )
        )
        or "tech"
    )

    client = LLMClient(
        endpoint=llm_cfg["endpoint"],
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 1024),
    )

    answer = client.generate(query, results, doc_type=doc_type)

    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print(f"Sources: {len(results)} chunk(s) retrieved")
    print("=" * 60)
    print(answer)
    if results:
        print("-" * 60)
        print("Retrieved chunks:")
        for i, c in enumerate(results, 1):
            print(f"  [{i}] {c.metadata.filename} ({c.metadata.source})")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Multi-Layer RAG System"
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Config file path"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Run ingestion pipeline")
    ingest_parser.set_defaults(func="ingest")

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query_text", help="Search query")
    query_parser.add_argument(
        "--doc-type",
        choices=["user", "tech", "support"],
        help="Override document type filter",
    )
    query_parser.set_defaults(func="query")

    chat_parser = subparsers.add_parser("chat", help="Start interactive REPL")
    chat_parser.set_defaults(func="chat")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.func == "ingest":
        cmd_ingest(config)
    elif args.func == "query":
        cmd_query(config, args.query_text, args.doc_type)
    elif args.func == "chat":
        from src.cli.repl import REPL
        repl = REPL(config)
        repl.run()


if __name__ == "__main__":
    main()
