"""Shared Chroma/hash/upsert for nutrition ingest (md + json)."""
from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nutrition_ingest")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

try:
    from src.shared.indexer import (
        get_or_create_vectorstore, compute_chunk_hash, compute_file_fingerprint,
        _resolve_index_db_dir)
    from src.shared.chunk_strategies import serialize_metatdata
    from src.shared.models.chunk import Chunk, ChunkMetadata
    from src.shared.metadata_store import (
        init_db, upsert_file_and_chunks, get_chunk_hashes_for_file, get_file_hash,
        find_vector_ids_for_chunk_hashes, delete_chunks_by_hashes)
    from src.shared.config import META_DB_PATH, EMBEDDING_MODEL
except ImportError:  # chạy từ thư mục scripts/
    from shared.indexer import (  # type: ignore
        get_or_create_vectorstore, compute_chunk_hash, compute_file_fingerprint,
        _resolve_index_db_dir)
    from shared.chunk_strategies import serialize_metatdata  # type: ignore
    from shared.models.chunk import Chunk, ChunkMetadata  # type: ignore
    from shared.metadata_store import (  # type: ignore
        init_db, upsert_file_and_chunks, get_chunk_hashes_for_file, get_file_hash,
        find_vector_ids_for_chunk_hashes, delete_chunks_by_hashes)
    from shared.config import META_DB_PATH, EMBEDDING_MODEL  # type: ignore

STRATEGY = "structure"
DATASET = "nutrition"
BATCH = 32
FOODS_DIR = ROOT / "data" / "raw" / "nutrition" / "foods"
NUTRITION_DIR = ROOT / "data" / "raw" / "nutrition"


def get_store():
    init_db(META_DB_PATH)
    return get_or_create_vectorstore(_resolve_index_db_dir(STRATEGY))


def prep(source_id: str, chunk_id: str, text: str, section_path: list,
         idx: int, extra: dict) -> tuple | None:
    """Build (text, meta, vid, row, hash) — None if hash already ingested."""
    chunk = Chunk(source_id=source_id, chunk_id=chunk_id, text=text,
                  metadata=ChunkMetadata(section_path=section_path, chunk_index=idx))
    h = compute_chunk_hash(chunk)
    if h in extra.pop("_known", set()):
        return None
    raw = chunk.metadata.model_dump()
    raw.update({"source_id": source_id, "chunking_strategy": STRATEGY,
                "dataset": DATASET, "embedding_model": EMBEDDING_MODEL,
                "chunk_hash": h, **extra})
    vid = f"{extra.get('file_name', source_id)}_{chunk_id}_{h[:12]}"
    row = {"chunk_hash": h, "chunk_index": idx, "vector_id": vid, "extra_meta": raw}
    return chunk.text, serialize_metatdata(raw), vid, row, h


def add_texts(db, texts, metas, ids) -> None:
    for i in range(0, len(texts), BATCH):
        sl = slice(i, i + BATCH)
        db.add_texts(texts=texts[sl], metadatas=metas[sl], ids=ids[sl])


def finish(file_key: str, fp_hash: str, rows: list) -> None:
    upsert_file_and_chunks(file_key, fp_hash, rows)


def run(src: str = "md"):
    """One path for both sources: --src md|json."""
    if src == "json":
        from ingest_food_json import main
    else:
        from ingest_nutrition import main
    return main()


if __name__ == "__main__":
    import sys

    _src = "md"
    if "--src" in sys.argv:
        _src = sys.argv[sys.argv.index("--src") + 1]
    elif "--json" in sys.argv:
        _src = "json"
    run(_src)
