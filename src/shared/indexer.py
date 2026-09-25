import os
import logging
import hashlib
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
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

pdf_options = PdfPipelineOptions()
pdf_options.do_ocr = False


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
    if EMBEDDING_PROVIDER == "openai":
        return str(BASE_DIR / "embeddings" / "pdf_db_openai" / strategy_value)
    return os.path.join(PDF_DB_DIR, strategy_value)


def get_or_create_vectorstore(db_dir: str):
    embeddings = _make_embeddings()
    return Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )


# =========================
# Reindex
# =========================
def hybrid_hash_reindex(pdf_path, vector_db, strategy, embedding_adapter, version_label: str | None = None, publication_date: str | None = None):
    fname = os.path.basename(pdf_path)
    file_key = f"{fname}::{strategy.value}"

    new_fp = compute_file_fingerprint(pdf_path)
    old_fp = get_file_hash(file_key)
    if new_fp == old_fp:
        logging.info(f"[SKIP] {fname}")
        return

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
        }
    )
    try:
        doc = converter.convert(pdf_path).document
        raw_text = extract_full_text_from_doc(doc)
        chunks = chunk_document(
            doc,
            pdf_path,
            strategy,
            raw_text=raw_text,
            embed_fn=embedding_adapter.embed_texts,
        )
    except Exception as e:
        # Docling chet (VD: std::bad_alloc tren PDF nang) -> fallback pymupdf.
        # SLIDING/SEMANTIC dung chung code path (chi can raw_text); HYBRID mimic theo trang.
        # STRUCTURE bat buoc docling (can headings) -> khong fallback duoc.
        from src.shared.chunk_strategies import ChunkingStrategy as _CS
        if strategy == _CS.STRUCTURE:
            raise
        logging.warning(f"[indexer] docling that bai ({e}), fallback pymupdf: {fname}")
        from src.shared.chunk_strategies import ChunkingStrategy as _CS
        pages = _pymupdf_pages(pdf_path)
        raw_text = "\n".join(pages)
        if not raw_text.strip():
            raise
        if strategy == _CS.HYBRID_SECTION_SEMANTIC:
            from src.shared.chunk_strategies import normalize_source_id
            chunks = _hybrid_fallback_pages(pages, normalize_source_id(pdf_path),
                                            embedding_adapter.embed_texts)
        else:
            chunks = chunk_document(
                None, pdf_path, strategy,
                raw_text=raw_text, embed_fn=embedding_adapter.embed_texts,
            )

    logging.info(
        f"[DEBUG] {fname} | {strategy.value} | chunked: {len(chunks)}")

    new_hashes = [compute_chunk_hash(c) for c in chunks]
    old_hashes = set(get_chunk_hashes_for_file(fname) or [])

    removed = old_hashes - set(new_hashes)
    added = [(i, h) for i, h in enumerate(new_hashes) if h not in old_hashes]

    if removed:
        vids = find_vector_ids_for_chunk_hashes(list(removed))
        vector_db.delete(ids=vids)
        delete_chunks_by_hashes(list(removed))

    rows, texts, metas, ids = [], [], [], []

    for i, h in added:
        c = chunks[i]
        vid = f"{fname}_{c.chunk_id}_{h[:12]}"

        raw_meta = c.metadata.model_dump()
        raw_meta.update({
            "source_id": c.source_id,
            "chunking_strategy": strategy.value
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
        f"[OK] {fname} | {strategy.value} | "
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
        from shared.chunk_strategies import ChunkingStrategy
    except ImportError:
        from src.shared.chunk_strategies import ChunkingStrategy

    try:
        from shared.config import INDEX_STRATEGIES
    except ImportError:
        from src.shared.config import INDEX_STRATEGIES
    _STRATS = {
        "structure": ChunkingStrategy.STRUCTURE,
        "sliding": ChunkingStrategy.SLIDING,
        "semantic": ChunkingStrategy.SEMANTIC,
        "hybrid_section_semantic": ChunkingStrategy.HYBRID_SECTION_SEMANTIC,
    }
    strategies = [_STRATS[name] for name in INDEX_STRATEGIES if name in _STRATS] or [ChunkingStrategy.STRUCTURE]
    for strategy in strategies:
        # Tạo thư mục vector DB riêng cho từng strategy (tách theo embedding provider)
        strategy_db_dir = _resolve_index_db_dir(strategy.value)
        os.makedirs(strategy_db_dir, exist_ok=True)

        vector_db = get_or_create_vectorstore(strategy_db_dir)

        # EmbeddingAdapter: nếu OpenAI, dùng OpenAIEmbeddings + tokenizer fake (không cần truncate HF)
        if EMBEDDING_PROVIDER == "openai" and _HAS_OPENAI_EMB:
            from langchain_openai import OpenAIEmbeddings as _OE
            embedder = _OE(model=OPENAI_EMBEDDING_MODEL)
            # tokenizer cho truncate: dùng tiktoken fallback, nếu không có thì dùng HF tokenizer
            try:
                import tiktoken
                enc = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL.replace("text-embedding-", "gpt-4"))
                class _TiktokenWrap:
                    def __call__(self, text, truncation=True, max_length=512, return_tensors=None):
                        toks = enc.encode(text)[:max_length]
                        return {"input_ids": toks}
                    def decode(self, ids, skip_special_tokens=True):
                        return enc.decode(ids)
                tokenizer = _TiktokenWrap()
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        else:
            embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

        embedding_adapter = EmbeddingAdapter(
            embedder=embedder,
            tokenizer=tokenizer,
            max_len=512,
        )

        for pdf_path in _iter_pdf_files(Path(PDF_DIR)):
            path = str(pdf_path)
            hybrid_hash_reindex(
                path,
                vector_db,
                strategy,
                embedding_adapter
            )


if __name__ == "__main__":
    main()
