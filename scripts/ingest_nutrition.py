"""
Ingest bảng dinh dưỡng curated (markdown) vào Chroma collection `structure`.

Vì indexer chính (`src/shared/indexer.py`) chỉ quét PDF, script này nạp file
`data/raw/nutrition/*.md` theo đúng schema metadata của indexer để retriever,
tracing và RAGAS dùng được ngay mà không cần sửa code RAG:
  - tách chunk theo heading `## ` (mỗi section là một chunk tự chứa)
  - metadata: source_id / section_path / chunk_index / dataset=nutrition /
    file_name / chunking_strategy=structure / embedding_model / chunk_hash
  - idempotent: chunk nào đã có (theo chunk_hash trong metadata_store) thì bỏ qua

Chạy:  python scripts/ingest_nutrition.py [--src md|json]
  (--src json = nạp foods JSON thay vì .md; cùng path với ingest_food_json.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "nutrition"))

from _ingest import (
    NUTRITION_DIR, STRATEGY, add_texts, compute_file_fingerprint, finish,
    get_chunk_hashes_for_file, get_file_hash, get_store, log, prep, run,
)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Tách markdown thành [(heading, body)]. Đoạn trước heading đầu tiên giữ lại."""
    sections: list[tuple[str, str]] = []
    heading: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if heading is not None or buf:
                sections.append((heading or "Mở đầu", "\n".join(buf).strip()))
            heading = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if heading is not None or buf:
        sections.append((heading or "Mở đầu", "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


def ingest_file(md_path: Path, vector_db) -> dict:
    fname = md_path.name
    file_key = f"{fname}::{STRATEGY}"
    source_id = md_path.stem.lower().replace(" ", "_")
    new_fp = compute_file_fingerprint(str(md_path))
    if get_file_hash(file_key) == new_fp:
        log.info("[SKIP] %s (fingerprint unchanged)", fname)
        return {"file": fname, "added": 0, "skipped": True}

    sections = split_sections(md_path.read_text(encoding="utf-8"))
    known = set(get_chunk_hashes_for_file(file_key) or [])
    texts, metas, ids, rows = [], [], [], []
    for idx, (heading, body) in enumerate(sections):
        out = prep(source_id, f"{source_id}_{idx}", f"{heading}\n{body}",
                   [heading], idx, {"file_name": fname, "_known": known})
        if out is None:
            continue
        texts.append(out[0])
        metas.append(out[1])
        ids.append(out[2])
        rows.append(out[3])
    if texts:
        add_texts(vector_db, texts, metas, ids)
    finish(file_key, new_fp, rows)
    log.info("[OK] %s | added=%d", fname, len(texts))
    return {"file": fname, "added": len(texts), "skipped": False}


def main() -> None:
    if "--src" in sys.argv and sys.argv[sys.argv.index("--src") + 1] == "json":
        return run("json")
    vector_db = get_store()
    files = sorted(NUTRITION_DIR.glob("*.md"))
    if not files:
        log.warning("Không có file .md nào trong %s", NUTRITION_DIR)
        return
    total = 0
    for f in files:
        total += ingest_file(f, vector_db)["added"]
    log.info("Done. total_added=%d", total)


if __name__ == "__main__":
    main()
