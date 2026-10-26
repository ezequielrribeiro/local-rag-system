"""Manifesto de hashes para ingestão incremental (CHG-R04)."""

import hashlib
import json
import os

MANIFEST_FILENAME = "ingest-manifest.json"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(processed_dir: str) -> str:
    return os.path.join(processed_dir, MANIFEST_FILENAME)


def load_manifest(processed_dir: str) -> dict:
    path = manifest_path(processed_dir)
    if not os.path.isfile(path):
        return {"files": {}}

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"files": {}}

    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        return {"files": {}}

    return data


def save_manifest(processed_dir: str, files: dict) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    with open(manifest_path(processed_dir), "w", encoding="utf-8") as fh:
        json.dump({"files": files}, fh, ensure_ascii=False, indent=2, sort_keys=True)


def current_hashes(raw_dir: str) -> dict:
    from src.ingestion.pipeline import iter_source_files

    result = {}
    for file_path, _doc_type, _chunker_cls in iter_source_files(raw_dir):
        result[file_path] = {"sha256": sha256_file(file_path)}
    return result


def plan(raw_dir: str, processed_dir: str) -> tuple[list[str], list[str], dict]:
    """Devolve (arquivos alterados/novos, arquivos removidos, hashes atuais)."""
    previous = (load_manifest(processed_dir).get("files") or {})
    current = current_hashes(raw_dir)

    changed = sorted(
        path
        for path, meta in current.items()
        if path not in previous or previous[path].get("sha256") != meta["sha256"]
    )
    removed = sorted(path for path in previous if path not in current)

    return changed, removed, current
