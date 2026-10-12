"""
Migrate vector DBs sang local (384d, all-MiniLM-L6-v2) hoặc NVIDIA (2048d).

  embeddings/pdf_db_openai/{4 strategies} -> embeddings/pdf_db[_nvidia]/{...}
  embeddings/memory_db_openai -> embeddings/memory_db[_nvidia]

Chạy:  python scripts/reembed.py --provider local|nvidia
Compat: scripts/reembed_local.py / reembed_nvidia.py (shim, giữ CLI cũ).
"""
from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import chromadb  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402

STRATEGIES = ["structure", "sliding", "semantic", "hybrid_section_semantic"]
# provider -> (dst_suffix, batch, wipe_pdf_dst, keep_legacy_memory)
PROVIDERS = {
    "local": ("pdf_db", 64, True, True),
    "nvidia": ("pdf_db_nvidia", 32, False, False),
}


def _base() -> Path:
    try:
        from src.shared.config import BASE_DIR
        return Path(str(BASE_DIR))
    except ImportError:
        return Path(__file__).resolve().parents[1]


def _embedder(provider: str):
    if provider == "nvidia":
        try:
            from src.shared.nvidia import NvidiaEmbeddings
            from src.shared.config import NVIDIA_EMBEDDING_MODEL
        except ImportError:
            from shared.nvidia import NvidiaEmbeddings  # type: ignore
            from shared.config import NVIDIA_EMBEDDING_MODEL  # type: ignore
        log.info("embedding model: %s", NVIDIA_EMBEDDING_MODEL)
        return NvidiaEmbeddings(model=NVIDIA_EMBEDDING_MODEL)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


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


def _add_batch(dstore, embedder, data, batch: int, tag: str) -> None:
    done = 0
    for i in range(0, len(data["ids"]), batch):
        sl = slice(i, i + batch)
        vecs = embedder.embed_documents(data["documents"][sl])
        dstore._collection.add(
            ids=data["ids"][sl], documents=data["documents"][sl],
            metadatas=data["metadatas"][sl], embeddings=vecs)
        done += len(data["ids"][sl])
        log.info("... %s %d/%d", tag, done, len(data["ids"]))


def migrate_pdf(embedder, provider: str) -> None:
    base, (dst_name, batch, wipe, _) = _base(), PROVIDERS[provider]
    for strat in STRATEGIES:
        src = base / "embeddings" / "pdf_db_openai" / strat
        dst = base / "embeddings" / dst_name / strat
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
        if wipe:
            import shutil
            if dst.exists():
                shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        dstore = Chroma(persist_directory=str(dst), embedding_function=embedder)
        _add_batch(dstore, embedder, data, batch, strat)
        log.info("[OK] pdf %s: migrated=%d", strat, len(data["ids"]))


def migrate_memory(embedder, provider: str) -> None:
    base, (_, batch, _, keep_legacy) = _base(), PROVIDERS[provider]
    dst_name = "memory_db" if provider == "local" else "memory_db_nvidia"
    src = base / "embeddings" / "memory_db_openai"
    dst = base / "embeddings" / dst_name
    legacy = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    if keep_legacy:
        try:
            lcol = _col(dst)
            if lcol.count():
                r = lcol.get(limit=lcol.count(),
                             include=["documents", "metadatas", "embeddings"])
                legacy = {"ids": r["ids"], "documents": r["documents"],
                          "metadatas": r["metadatas"], "embeddings": r["embeddings"]}
                log.info("memory legacy: giữ %d facts", len(legacy["ids"]))
        except Exception as e:
            log.info("memory legacy: %s", str(e)[:100])
    try:
        data = _read_all(_col(src))
    except Exception as e:
        log.warning("[SKIP] memory openai: %s", str(e)[:120])
        return
    if not keep_legacy and _dst_count(dst) == len(data["ids"]) and data["ids"]:
        log.info("[SKIP] memory: đích đã đủ %d", len(data["ids"]))
        return
    if keep_legacy:  # Windows lock file: xóa collection qua client, không rmtree
        try:
            chromadb.PersistentClient(path=str(dst)).delete_collection("langchain")
        except Exception:
            pass
    else:
        dst.mkdir(parents=True, exist_ok=True)
    dstore = Chroma(persist_directory=str(dst), embedding_function=embedder)
    if legacy["ids"]:
        dstore._collection.add(ids=legacy["ids"], documents=legacy["documents"],
                               metadatas=legacy["metadatas"],
                               embeddings=legacy["embeddings"])
    _add_batch(dstore, embedder, data, batch, "memory")
    log.info("[OK] memory: legacy=%d migrated=%d", len(legacy["ids"]), len(data["ids"]))


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Re-embed vector DBs (local|nvidia)")
    p.add_argument("--provider", default="local", choices=["local", "nvidia"])
    args = p.parse_args(argv)
    embedder = _embedder(args.provider)
    migrate_pdf(embedder, args.provider)
    migrate_memory(embedder, args.provider)
    log.info("Done.")


if __name__ == "__main__":
    main()
