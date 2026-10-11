"""
Ingest bảng dinh dưỡng curated (markdown) vào Chroma collection `structure`.

Vì indexer chính (`src/shared/indexer.py`) chỉ quét PDF, script này nạp file
`data/raw/nutrition/*.md` theo đúng schema metadata của indexer để retriever,
tracing và RAGAS dùng được ngay mà không cần sửa code RAG:
  - tách chunk theo heading `## ` (mỗi section là một chunk tự chứa)
  - metadata: source_id / section_path / chunk_index / dataset=nutrition /
    file_name / chunking_strategy=structure / embedding_model / chunk_hash
  - idempotent: chunk nào đã có (theo chunk_hash trong metadata_store) thì bỏ qua

Chạy:  python scripts/ingest_nutrition.py
"""
from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    from src.shared.indexer import (
        get_or_create_vectorstore,
        compute_chunk_hash,
        compute_file_fingerprint,
        _resolve_index_db_dir,
    )
    from src.shared.chunk_strategies import serialize_metatdata
    from src.shared.models.chunk import Chunk, ChunkMetadata
    from src.shared.metadata_store import (
        init_db,
        upsert_file_and_chunks,
        get_chunk_hashes_for_file,
        get_file_hash,
    )
    from src.shared.config import BASE_DIR, META_DB_PATH, EMBEDDING_MODEL
except ImportError:  # chạy từ thư mục scripts/
    from shared.indexer import (  # type: ignore
        get_or_create_vectorstore,
        compute_chunk_hash,
        compute_file_fingerprint,
        _resolve_index_db_dir,
    )
    from shared.chunk_strategies import serialize_metatdata  # type: ignore
    from shared.models.chunk import Chunk, ChunkMetadata  # type: ignore
    from shared.metadata_store import (  # type: ignore
        init_db,
        upsert_file_and_chunks,
        get_chunk_hashes_for_file,
        get_file_hash,
    )
    from shared.config import BASE_DIR, META_DB_PATH, EMBEDDING_MODEL  # type: ignore

NUTRITION_DIR = Path(str(BASE_DIR)) / "data" / "raw" / "nutrition"
STRATEGY = "structure"
DATASET = "nutrition"


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
    old_hashes = set(get_chunk_hashes_for_file(file_key) or [])
    texts, metas, ids, rows = [], [], [], []
    for idx, (heading, body) in enumerate(sections):
        chunk = Chunk(
            source_id=source_id,
            chunk_id=f"{source_id}_{idx}",
            text=f"{heading}\n{body}",
            metadata=ChunkMetadata(section_path=[heading], chunk_index=idx),
        )
        h = compute_chunk_hash(chunk)
        if h in old_hashes:
            continue
        vid = f"{fname}_{chunk.chunk_id}_{h[:12]}"
        raw_meta = chunk.metadata.model_dump()
        raw_meta.update({
            "source_id": source_id,
            "chunking_strategy": STRATEGY,
            "dataset": DATASET,
            "file_name": fname,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_hash": h,
        })
        texts.append(chunk.text)
        metas.append(serialize_metatdata(raw_meta))
        ids.append(vid)
        rows.append({
            "chunk_hash": h,
            "chunk_index": idx,
            "vector_id": vid,
            "extra_meta": raw_meta,
        })
    if texts:
        vector_db.add_texts(texts=texts, metadatas=metas, ids=ids)
    upsert_file_and_chunks(file_key, new_fp, rows)
    log.info("[OK] %s | added=%d", fname, len(texts))
    return {"file": fname, "added": len(texts), "skipped": False}


def main() -> None:
    init_db(META_DB_PATH)
    vector_db = get_or_create_vectorstore(_resolve_index_db_dir(STRATEGY))
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
