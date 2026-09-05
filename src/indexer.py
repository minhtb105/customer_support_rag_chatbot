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
    from chunk_strategies import (
        chunk_document,
        extract_full_text_from_doc,
        serialize_metatdata
    )
    from embedding.adapter import EmbeddingAdapter
    from models.chunk import Chunk
    from metadata_store import *
    from config import *
except ImportError:
    from src.chunk_strategies import (
        chunk_document,
        extract_full_text_from_doc,
        serialize_metatdata
    )
    from src.embedding.adapter import EmbeddingAdapter
    from src.models.chunk import Chunk
    from src.metadata_store import *
    from src.config import *


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
        h.update(f.read(sample_size))
        if stat.st_size > sample_size:
            f.seek(max(0, stat.st_size - sample_size))
            h.update(f.read(sample_size))

    return h.hexdigest()


def _make_embeddings():
    if EMBEDDING_PROVIDER == "openai" and _HAS_OPENAI_EMB:
        try:
            return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        except Exception as e:
            logging.warning(f"[indexer] OpenAIEmbeddings fail ({e}), fallback HuggingFace")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


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
def hybrid_hash_reindex(pdf_path, vector_db, strategy, embedding_adapter):
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
    doc = converter.convert(pdf_path).document
    raw_text = extract_full_text_from_doc(doc)

    chunks = chunk_document(
        doc,
        pdf_path,
        strategy,
        raw_text=raw_text,
        embed_fn=embedding_adapter.embed_texts,
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
        from chunk_strategies import ChunkingStrategy
    except ImportError:
        from src.chunk_strategies import ChunkingStrategy

    for strategy in [
        ChunkingStrategy.STRUCTURE,
        ChunkingStrategy.SLIDING,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.HYBRID_SECTION_SEMANTIC,
    ]:
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
