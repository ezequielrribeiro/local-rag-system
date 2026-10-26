import os

from src.ingestion import incremental


def test_plan_detects_new_changed_and_removed(tmp_path):
    raw = tmp_path / "raw"
    (raw / "tech").mkdir(parents=True)
    processed = tmp_path / "processed"
    processed.mkdir()

    file = raw / "tech" / "a.md"
    file.write_text("# A\nconteudo", encoding="utf-8")

    changed, removed, current = incremental.plan(str(raw), str(processed))
    assert str(file) in changed
    assert removed == []

    incremental.save_manifest(str(processed), current)
    changed, removed, _ = incremental.plan(str(raw), str(processed))
    assert changed == []
    assert removed == []

    file.write_text("# A\nconteudo novo", encoding="utf-8")
    changed, removed, _ = incremental.plan(str(raw), str(processed))
    assert str(file) in changed

    incremental.save_manifest(str(processed), incremental.current_hashes(str(raw)))
    file.unlink()
    changed, removed, _ = incremental.plan(str(raw), str(processed))
    assert str(file) in removed


def test_sha256_file_is_stable(tmp_path):
    file = tmp_path / "x.txt"
    file.write_text("abc", encoding="utf-8")
    first = incremental.sha256_file(str(file))
    second = incremental.sha256_file(str(file))
    assert first == second
    assert len(first) == 64
