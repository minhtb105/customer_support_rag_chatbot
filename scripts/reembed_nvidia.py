"""
Migrate vector DBs sang NVIDIA embeddings (nemotron-3-embed-1b, 2048d).

  embeddings/pdf_db_openai/{4 strategies} -> embeddings/pdf_db_nvidia/{...}
      (giữ nguyên ids + metadatas, re-embed passage mode)
  embeddings/memory_db_openai -> embeddings/memory_db_nvidia
      (giữ facts legacy nvidia nếu có, thêm facts migrate)

Idempotent theo count: đích đã đủ số docs thì SKIP.
Xoay key 401/429/503 qua src/shared/nvidia.py.
Chạy:  python scripts/reembed_nvidia.py
"""
from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import chromadb  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402

try:
    from src.shared.nvidia import NvidiaEmbeddings
    from src.shared.config import NVIDIA_EMBEDDING_MODEL, BASE_DIR
except ImportError:
    from shared.nvidia import NvidiaEmbeddings  # type: ignore
    from shared.config import NVIDIA_EMBEDDING_MODEL, BASE_DIR  # type: ignore

STRATEGIES = ["structure", "sliding", "semantic", "hybrid_section_semantic"]
BATCH = 32


def _col(path: Path):
    return chromadb.PersistentClient(path=str(path)).get_collection("langchain")


def _read_all(col):
    n = col.count()
    if not n:
        return {"ids": [], "documents": [], "metadatas": []}
    out = col.get(limit=n, include=["documents", "metadatas"])
    return {"ids": out["ids"], "documents": out["documents"],
            "metadatas": out["metadatas"]}


def _dst_count(path: Path) -> int:
    try:
        return _col(path).count()
    except Exception:
        return 0


def migrate_pdf(embedder: NvidiaEmbeddings) -> None:
    for strat in STRATEGIES:
        src = Path(str(BASE_DIR)) / "embeddings" / "pdf_db_openai" / strat
        dst = Path(str(BASE_DIR)) / "embeddings" / "pdf_db_nvidia" / strat
        try:
            data = _read_all(_col(src))
        except Exception as e:
            log.warning("[SKIP] %s: %s", strat, str(e)[:120])
            continue
        if not data["ids"]:
            log.info("[SKIP] %s: nguồn rỗng", strat)
            continue
        if _dst_count(dst) == len(data["ids"]):
            log.info("[SKIP] %s: đích đã đủ %d", strat, len(data["ids"]))
            continue
        dst.mkdir(parents=True, exist_ok=True)
        dstore = Chroma(persist_directory=str(dst), embedding_function=embedder)
        done = 0
        for i in range(0, len(data["ids"]), BATCH):
            sl = slice(i, i + BATCH)
            vecs = embedder.embed_documents(data["documents"][sl])
            dstore._collection.add(
                ids=data["ids"][sl], documents=data["documents"][sl],
                metadatas=data["metadatas"][sl], embeddings=vecs)
            done += len(data["ids"][sl])
            log.info("... %s %d/%d", strat, done, len(data["ids"]))
        log.info("[OK] pdf %s: migrated=%d", strat, len(data["ids"]))


def migrate_memory(embedder: NvidiaEmbeddings) -> None:
    src = Path(str(BASE_DIR)) / "embeddings" / "memory_db_openai"
    dst = Path(str(BASE_DIR)) / "embeddings" / "memory_db_nvidia"
    try:
        data = _read_all(_col(src))
    except Exception as e:
        log.warning("[SKIP] memory openai: %s", str(e)[:120])
        return
    if _dst_count(dst) == len(data["ids"]) and data["ids"]:
        log.info("[SKIP] memory: đích đã đủ %d", len(data["ids"]))
        return
    dst.mkdir(parents=True, exist_ok=True)
    dstore = Chroma(persist_directory=str(dst), embedding_function=embedder)
    for i in range(0, len(data["ids"]), BATCH):
        sl = slice(i, i + BATCH)
        vecs = embedder.embed_documents(data["documents"][sl])
        dstore._collection.add(
            ids=data["ids"][sl], documents=data["documents"][sl],
            metadatas=data["metadatas"][sl], embeddings=vecs)
    log.info("[OK] memory: migrated=%d", len(data["ids"]))


def main() -> None:
    embedder = NvidiaEmbeddings(model=NVIDIA_EMBEDDING_MODEL)
    log.info("embedding model: %s", NVIDIA_EMBEDDING_MODEL)
    migrate_pdf(embedder)
    migrate_memory(embedder)
    log.info("Done.")


if __name__ == "__main__":
    main()
