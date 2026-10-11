"""
Migrate vector DBs từ OpenAI embeddings (1536d) sang local (384d, all-MiniLM-L6-v2).

  embeddings/pdf_db_openai/{structure,sliding,semantic,hybrid_section_semantic}
      -> embeddings/pdf_db/{...}  (wipe + re-embed, giữ nguyên ids + metadatas)
  embeddings/memory_db_openai -> embeddings/memory_db
      (giữ lại facts legacy 384d đã có, thêm facts migrate re-embed)

Idempotent: chạy lại sẽ rebuild lại từ nguồn openai (nguồn không đổi nên kết quả giống nhau).
Chạy:  python scripts/reembed_local.py
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE = Path(__file__).resolve().parents[1]
STRATEGIES = ["structure", "sliding", "semantic", "hybrid_section_semantic"]
BATCH = 64


def _wipe(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def _read_all(col):
    n = col.count()
    if not n:
        return {"ids": [], "documents": [], "metadatas": []}
    out = col.get(limit=n, include=["documents", "metadatas"])
    return {"ids": out["ids"], "documents": out["documents"],
            "metadatas": out["metadatas"]}


def migrate_pdf() -> None:
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    for strat in STRATEGIES:
        src = BASE / "embeddings" / "pdf_db_openai" / strat
        dst = BASE / "embeddings" / "pdf_db" / strat
        try:
            scol = chromadb.PersistentClient(path=str(src)).get_collection("langchain")
        except Exception as e:
            log.warning("[SKIP] %s: %s", strat, str(e)[:120])
            continue
        data = _read_all(scol)
        if not data["ids"]:
            log.info("[SKIP] %s: nguồn rỗng", strat)
            continue
        try:
            dst_n = chromadb.PersistentClient(
                path=str(dst)).get_collection("langchain").count()
        except Exception:
            dst_n = 0
        if dst_n == len(data["ids"]):
            log.info("[SKIP] %s: đích đã đủ %d", strat, dst_n)
            continue
        _wipe(dst)
        dstore = Chroma(persist_directory=str(dst), embedding_function=embedder)
        for i in range(0, len(data["ids"]), BATCH):
            sl = slice(i, i + BATCH)
            vecs = embedder.embed_documents(data["documents"][sl])
            dstore._collection.add(
                ids=data["ids"][sl], documents=data["documents"][sl],
                metadatas=data["metadatas"][sl], embeddings=vecs)
        log.info("[OK] pdf %s: migrated=%d", strat, len(data["ids"]))


def migrate_memory() -> None:
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    src = BASE / "embeddings" / "memory_db_openai"
    dst = BASE / "embeddings" / "memory_db"
    # 1. giữ facts legacy 384d (kèm embeddings gốc)
    legacy = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    try:
        lcol = chromadb.PersistentClient(path=str(dst)).get_collection("langchain")
        n = lcol.count()
        if n:
            r = lcol.get(limit=n, include=["documents", "metadatas", "embeddings"])
            legacy = {"ids": r["ids"], "documents": r["documents"],
                      "metadatas": r["metadatas"], "embeddings": r["embeddings"]}
            log.info("memory legacy: giữ %d facts", n)
    except Exception as e:
        log.info("memory legacy: %s", str(e)[:100])
    # 2. đọc nguồn openai
    try:
        scol = chromadb.PersistentClient(path=str(src)).get_collection("langchain")
    except Exception as e:
        log.warning("[SKIP] memory openai: %s", str(e)[:120])
        return
    data = _read_all(scol)
    # Xóa collection qua client (Windows lock file, không rmtree được khi client còn mở)
    dclient = chromadb.PersistentClient(path=str(dst))
    try:
        dclient.delete_collection("langchain")
    except Exception:
        pass
    dstore = Chroma(persist_directory=str(dst), embedding_function=embedder)
    if legacy["ids"]:
        dstore._collection.add(ids=legacy["ids"], documents=legacy["documents"],
                              metadatas=legacy["metadatas"],
                              embeddings=legacy["embeddings"])
    for i in range(0, len(data["ids"]), BATCH):
        sl = slice(i, i + BATCH)
        vecs = embedder.embed_documents(data["documents"][sl])
        dstore._collection.add(
            ids=data["ids"][sl], documents=data["documents"][sl],
            metadatas=data["metadatas"][sl], embeddings=vecs)
    log.info("[OK] memory: legacy=%d migrated=%d total=%d",
             len(legacy["ids"]), len(data["ids"]),
             len(legacy["ids"]) + len(data["ids"]))


def main() -> None:
    migrate_pdf()
    migrate_memory()
    log.info("Done.")


if __name__ == "__main__":
    main()
