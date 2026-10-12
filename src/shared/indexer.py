import os
import logging
import hashlib
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from transformers import AutoTokenizer
try:
    from langchain_openai import OpenAIEmbeddings
    _HAS_OPENAI_EMB = True
except ImportError:
    _HAS_OPENAI_EMB = False
try:
    from shared.chunk_strategies import (
        chunk_document,
        extract_full_text_from_doc,
        serialize_metatdata,
        semantic_chunk,
    )
    from shared.embedding.adapter import EmbeddingAdapter
    from shared.models.chunk import Chunk, ChunkMetadata
    from shared.metadata_store import *
    from shared.config import *
except ImportError:
    from src.shared.chunk_strategies import (
        chunk_document,
        extract_full_text_from_doc,
        serialize_metatdata,
        semantic_chunk,
    )
    from src.shared.embedding.adapter import EmbeddingAdapter
    from src.shared.models.chunk import Chunk, ChunkMetadata
    from src.shared.metadata_store import *
    from src.shared.config import *


def _pymupdf_pages(pdf_path: str):
    """Fallback text theo trang khi docling chet (VD: bad_alloc)."""
    import fitz
    doc = fitz.open(pdf_path)
    try:
        return [p.get_text() or "" for p in doc]
    finally:
        doc.close()


def _no_docling(fname: str, big: bool) -> bool:
    """Single branch for the no-docling (pymupdf) path: skip-list or big PDF."""
    if fname in DOCLING_SKIP or big:
        logging.warning(f"[indexer] no-docling path (skip-list/big-PDF): {fname}")
        return True
    return False


def _hybrid_fallback_pages(pages, source_id: str, embed_fn):
    """Mimic hybrid_section_semantic theo tung trang (section=[page N])."""
    try:
        from shared.chunk_strategies import tokenizer as _tok
    except ImportError:
        from src.shared.chunk_strategies import tokenizer as _tok  # type: ignore
    chunks, idx = [], 0
    for n, text in enumerate(pages, start=1):
        text = (text or "").strip()
        if not text:
            continue
        if len(_tok.tokenizer.tokenize(text)) <= MAX_TOKENS:
            chunks.append(Chunk(source_id=source_id, chunk_id=f"{source_id}_{idx}",
                               text=text, metadata=ChunkMetadata(
                                   section_path=[f"page {n}"], page_numbers=[n], chunk_index=idx)))
            idx += 1
            continue
        for sub in semantic_chunk(text, source_id, embed_fn=embed_fn,
                                  atomic_tokenizer=_tok.tokenizer, atomic_size=120):
            sub.metadata.section_path = [f"page {n}"]
            sub.metadata.page_numbers = [n]
            sub.metadata.chunk_index = idx
            sub.chunk_id = f"{source_id}_{idx}"
            chunks.append(sub)
            idx += 1
    return chunks


logging.basicConfig(level=logging.INFO)


# =========================
# Vectorstore utils
# =========================
def compute_chunk_hash(chunk: Chunk) -> str:
    h = hashlib.sha256()
    h.update(" ".join(chunk.text.split()).encode())
    meta = chunk.metadata
    h.update(str(meta.section_path or "").encode())
    h.update(str(meta.page_numbers or "").encode())
    h.update(str(meta.chunk_index).encode())

    return h.hexdigest()


def compute_file_fingerprint(path: str, sample_size=512 * 512) -> str:
    stat = os.stat(path)
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime_ns).encode())

    with open(path, "rb") as f:
        # head
        h.update(f.read(sample_size))
        if stat.st_size > sample_size:
            # middle sample (catches middle-only changes, mitigates LLM09 Vector Weakness)
            middle = max(0, stat.st_size // 2 - 32768)
            try:
                f.seek(middle)
                h.update(f.read(65536))  # 64KB middle
            except Exception:
                pass
            # tail
            f.seek(max(0, stat.st_size - sample_size))
            h.update(f.read(sample_size))

    return h.hexdigest()


def _make_embeddings():
    if EMBEDDING_PROVIDER == "nvidia":
        from src.shared.nvidia import NvidiaEmbeddings
        from src.shared.config import NVIDIA_EMBEDDING_MODEL
        return NvidiaEmbeddings(model=NVIDIA_EMBEDDING_MODEL)
    if EMBEDDING_PROVIDER == "openai" and _HAS_OPENAI_EMB:
        try:
            return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        except Exception as e:
            logging.warning(f"[indexer] OpenAIEmbeddings fail ({e}), fallback HuggingFace")
    # Fallback LUÔN dùng model local (EMBEDDING_MODEL có thể là tên model OpenAI)
    from src.shared.config import EMBEDDING_DIMENSIONS_MAP
    local_model = EMBEDDING_MODEL if EMBEDDING_MODEL in ("all-MiniLM-L6-v2",) or EMBEDDING_DIMENSIONS_MAP.get(EMBEDDING_MODEL) == 384 else "all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=local_model)


def _resolve_index_db_dir(strategy_value: str) -> str:
    if EMBEDDING_PROVIDER == "nvidia":
        return str(BASE_DIR / "embeddings" / "pdf_db_nvidia" / strategy_value)
    if EMBEDDING_PROVIDER == "openai":
        return str(BASE_DIR / "embeddings" / "pdf_db_openai" / strategy_value)
    return os.path.join(PDF_DB_DIR, strategy_value)


def get_or_create_vectorstore(db_dir: str):
    return Chroma(persist_directory=db_dir, embedding_function=_make_embeddings())


class _TiktokenTokenizer:
    """HF-tokenizer-shaped wrapper over tiktoken (OpenAI truncate path)."""

    def __init__(self, enc):
        self._enc = enc

    def __call__(self, text, truncation=True, max_length=512, return_tensors=None):
        return {"input_ids": self._enc.encode(text)[:max_length]}

    def decode(self, ids, skip_special_tokens=True):
        return self._enc.decode(ids)


def _make_truncate_tokenizer():
    """Reuse _make_embeddings' provider choice; tiktoken for OpenAI, else HF."""
    if EMBEDDING_PROVIDER == "openai" and _HAS_OPENAI_EMB:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL.replace("text-embedding-", "gpt-4"))
            return _TiktokenTokenizer(enc)
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


# =========================
# Reindex
# =========================
def _chunk_via_pymupdf(pdf_path, strategy, overrides, embedding_adapter):
    """Docling-free path (crash escape hatch + docling-dead fallback)."""
    try:
        from shared.chunk_strategies import normalize_source_id, chunk_document as _cd2, ChunkingStrategy as _CS2
    except ImportError:
        from src.shared.chunk_strategies import normalize_source_id, chunk_document as _cd2, ChunkingStrategy as _CS2  # type: ignore
    pages = _pymupdf_pages(pdf_path)
    raw_text = "\n".join(pages)
    if not raw_text.strip():
        raise ValueError(f"pymupdf extracted no text: {pdf_path}")
    if strategy in (_CS2.HYBRID_SECTION_SEMANTIC, _CS2.STRUCTURE):
        # Mimic theo trang (section=[page N]); trang dài split semantic.
        # STRUCTURE cũng đi đường này khi docling chết (mất headings doc,
        # còn text + page citations) thay vì abort cả process.
        return _hybrid_fallback_pages(pages, normalize_source_id(pdf_path),
                                      embedding_adapter.embed_texts)
    return _cd2(None, pdf_path, strategy, raw_text=raw_text,
                embed_fn=embedding_adapter.embed_texts, **overrides)


# PDF nặng (nhiều trang) làm docling chết process-wide (std::bad_alloc /
# segfault, không catch được — đã thử batch trang + images_scale 0.5, vẫn
# chết) -> preselect theo page_count, không đợi crash: mọi strategy đi
# pymupdf (STRUCTURE mimic section=[page N], giữ text + page citations).
BIG_PDF_PAGES = int(os.getenv("INDEXER_BIG_PDF_PAGES", "100"))
# Trang độc lẻ (vd WHO_Pharmacological_Hypertension_2021.pdf 61 trang, chết ở
# trang 58-61): process abort nên chỉ phát hiện qua log — thêm basename vào
# env: INDEXER_DOCLING_SKIP="a.pdf,b.pdf" để đi pymupdf như PDF nặng.
DOCLING_SKIP = {s.strip() for s in os.getenv("INDEXER_DOCLING_SKIP", "").split(",") if s.strip()}


def _pdf_page_count(pdf_path: str) -> int:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        try:
            return doc.page_count
        finally:
            doc.close()
    except Exception:
        return 0


# Docling convert chạy trong child process riêng: bad_alloc/segfault chỉ giết
# child (exit != 0) -> parent đi pymupdf fallback, cả run không chết. Cần vì
# leak native: run dài convert tại chỗ chết cả ở file trước đó convert OK.
CONVERT_TIMEOUT_S = int(os.getenv("INDEXER_CONVERT_TIMEOUT_S", "1200"))


def _convert_docling_isolated(pdf_path: str):
    """Convert docling cách ly process. Trả DoclingDocument hoặc None (chết)."""
    import subprocess
    import sys as _sys
    import tempfile
    fd, tmp_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        worker = str(BASE_DIR / "scripts" / "docling_convert_worker.py")
        p = subprocess.run(
            [_sys.executable, worker, pdf_path, tmp_json],
            capture_output=True, text=True, timeout=CONVERT_TIMEOUT_S,
        )
        if p.returncode != 0 or not os.path.exists(tmp_json) or os.path.getsize(tmp_json) == 0:
            tail = (p.stderr or "")[-300:]
            logging.warning(f"[indexer] docling child died rc={p.returncode}: "
                            f"{os.path.basename(pdf_path)} {tail!r}")
            return None
        from docling_core.types.doc import DoclingDocument
        return DoclingDocument.load_from_json(tmp_json)
    except subprocess.TimeoutExpired:
        logging.warning(f"[indexer] docling child timeout: {os.path.basename(pdf_path)}")
        return None
    finally:
        try:
            os.remove(tmp_json)
        except OSError:
            pass


# Sentinel: caller không đưa _doc -> hybrid tự convert cách ly (đường monitor).
_AUTO_DOC = object()


def hybrid_hash_reindex(pdf_path, vector_db, strategy_name: str, embedding_adapter, version_label: str | None = None, publication_date: str | None = None, _doc=_AUTO_DOC):
    try:
        from shared.chunk_strategies import resolve_strategy, chunk_document as _cd, ChunkingStrategy as _CS0
    except ImportError:
        from src.shared.chunk_strategies import resolve_strategy, chunk_document as _cd, ChunkingStrategy as _CS0  # type: ignore
    strategy, overrides = resolve_strategy(strategy_name)  # fail LOUD on unknown
    fname = os.path.basename(pdf_path)
    # Provider-scoped key: cùng PDF index dưới openai/nvidia là 2 DB/không gian
    # vector khác nhau (dim khác nhau). Key mù-provider cũ gây SKIP nhầm chéo.
    file_key = f"{fname}::{EMBEDDING_PROVIDER}::{strategy_name}"

    new_fp = compute_file_fingerprint(pdf_path)
    old_fp = get_file_hash(file_key)
    if new_fp == old_fp:
        logging.info(f"[SKIP] {fname}")
        return

    # Convert: predoc từ main (1 convert/PDF cho mọi strategy) hoặc tự convert
    # cách ly khi caller đơn lẻ (monitor). None = đi pymupdf toàn phần.
    _big = _pdf_page_count(pdf_path) > BIG_PDF_PAGES
    if _doc is _AUTO_DOC:
        if os.getenv("INDEXER_PYMUPDF_ONLY") and strategy != _CS0.STRUCTURE:
            doc = None
        elif _no_docling(fname, _big):
            doc = None
        else:
            doc = _convert_docling_isolated(pdf_path)
    else:
        doc = _doc
    if doc is None:
        # Docling chết/không dùng: pymupdf cho MỌI strategy (STRUCTURE mimic
        # section=[page N], giữ text + page citations).
        if _doc is _AUTO_DOC:
            logging.warning(f"[indexer] docling failed, fallback pymupdf: {fname}")
        chunks = _chunk_via_pymupdf(pdf_path, strategy, overrides, embedding_adapter)
    else:
        raw_text = extract_full_text_from_doc(doc)
        chunks = _cd(
            doc,
            pdf_path,
            strategy,
            raw_text=raw_text,
            embed_fn=embedding_adapter.embed_texts,
            **overrides,
        )

    logging.info(
        f"[DEBUG] {fname} | {strategy_name} | chunked: {len(chunks)}")

    new_hashes = [compute_chunk_hash(c) for c in chunks]
    old_hashes = set(get_chunk_hashes_for_file(file_key) or [])

    removed = old_hashes - set(new_hashes)
    added = [(i, h) for i, h in enumerate(new_hashes) if h not in old_hashes]

    if removed:
        vids = find_vector_ids_for_chunk_hashes(list(removed))
        vector_db.delete(ids=vids)
        delete_chunks_by_hashes(list(removed))

    rows, texts, metas, ids = [], [], [], []

    for i, h in added:
        c = chunks[i]
        if not (c.text or "").strip():
            continue  # chunk trắng: NVIDIA 400 + vector rác — bỏ khỏi DB lẫn bookkeeping
        vid = f"{fname}_{c.chunk_id}_{h[:12]}"

        raw_meta = c.metadata.model_dump()
        raw_meta.update({
            "source_id": c.source_id,
            "chunking_strategy": strategy_name
        })
        if version_label:
            raw_meta["version_label"] = version_label
            raw_meta["version"] = version_label
        if publication_date:
            raw_meta["publication_date"] = publication_date

        meta = serialize_metatdata(raw_meta) 

        texts.append(c.text)
        metas.append(meta)
        ids.append(vid)

        rows.append({
            "chunk_hash": h,
            "chunk_index": c.metadata.chunk_index,
            "vector_id": vid,
            "extra_meta": raw_meta
        })

    if texts:
        vector_db.add_texts(texts=texts, metadatas=metas, ids=ids)

    upsert_file_and_chunks(file_key, new_fp, rows)

    logging.info(
        f"[OK] {fname} | {strategy_name} | "
        f"added={len(added)}, removed={len(removed)}"
    )


# =========================
# Main
# =========================
def _iter_pdf_files(root: Path):
    """Quét đệ quy PDF — hỗ trợ guideline subfolders (diabetes, who_iris, byt...)."""
    for p in root.rglob("*.pdf"):
        if p.is_file():
            yield p

def main():
    init_db(META_DB_PATH)

    try:
        from shared.chunk_strategies import VARIANTS, resolve_strategy
    except ImportError:
        from src.shared.chunk_strategies import VARIANTS, resolve_strategy

    try:
        from shared.config import INDEX_STRATEGIES
    except ImportError:
        from src.shared.config import INDEX_STRATEGIES
    strategies = []
    for name in INDEX_STRATEGIES:
        resolve_strategy(name)  # fail LOUD on unknown, never silent fallback
        strategies.append(name)
    if not strategies:
        strategies = ["structure"]
    vector_dbs = {}
    for strategy_name in strategies:
        # Tạo thư mục vector DB riêng cho từng strategy (tách theo embedding provider)
        strategy_db_dir = _resolve_index_db_dir(strategy_name)
        os.makedirs(strategy_db_dir, exist_ok=True)
        vector_dbs[strategy_name] = get_or_create_vectorstore(strategy_db_dir)

    # EmbeddingAdapter: reuse _make_embeddings (single provider branch);
    # truncate tokenizer: tiktoken for OpenAI, else HF.
    embedder = _make_embeddings()
    tokenizer = _make_truncate_tokenizer()

    embedding_adapter = EmbeddingAdapter(
        embedder=embedder,
        tokenizer=tokenizer,
        max_len=512,
    )

    # 1 convert/PDF cho mọi strategy (docling cách ly process; chết -> None ->
    # pymupdf). Vừa nhanh 5x, vừa hết leak native làm run dài chết giữa chừng.
    pymupdf_only = bool(os.getenv("INDEXER_PYMUPDF_ONLY"))
    for pdf_path in _iter_pdf_files(Path(PDF_DIR)):
        path = str(pdf_path)
        fname = os.path.basename(path)
        _big_pdf = _pdf_page_count(path) > BIG_PDF_PAGES
        if pymupdf_only or _no_docling(fname, _big_pdf):
            doc = None
        else:
            doc = _convert_docling_isolated(path)
            if doc is None:
                logging.warning(f"[indexer] docling failed, fallback pymupdf: {fname}")
        for strategy_name in strategies:
            hybrid_hash_reindex(
                path,
                vector_dbs[strategy_name],
                strategy_name,
                embedding_adapter,
                _doc=doc,
            )


if __name__ == "__main__":
    main()
