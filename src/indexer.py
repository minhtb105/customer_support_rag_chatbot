import os
import logging
import hashlib
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from transformers import AutoTokenizer
from chunk_strategies import (
    chunk_document,
    extract_full_text_from_doc,
)
from embedding.adapter import EmbeddingAdapter
from models.chunk import Chunk
from metadata_store import *
from config import *


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


def get_or_create_vectorstore(db_dir: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
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

        meta = c.metadata.model_dump()
        meta.update({
            "source_id": c.source_id,
            "chunking_strategy": strategy.value
        })

        texts.append(c.text)
        metas.append(meta)
        ids.append(vid)

        rows.append({
            "chunk_hash": h,
            "chunk_index": c.metadata.chunk_index,
            "vector_id": vid,
            "extra_meta": meta
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
def main():
    init_db(META_DB_PATH)

    from chunk_strategies import ChunkingStrategy

    for strategy in [
        ChunkingStrategy.STRUCTURE,
        ChunkingStrategy.SLIDING,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.HYBRID_SECTION_SEMANTIC,
    ]:
        # Tạo thư mục vector DB riêng cho từng strategy
        strategy_db_dir = os.path.join(PDF_DB_DIR, strategy.value)
        os.makedirs(strategy_db_dir, exist_ok=True)

        vector_db = get_or_create_vectorstore(strategy_db_dir)

        embedding_adapter = EmbeddingAdapter(
            embedder=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_MODEL),
            max_len=512,
        )

        for pdf in os.listdir(PDF_DIR):
            if not pdf.lower().endswith(".pdf"):
                continue
            path = os.path.join(PDF_DIR, pdf)

            hybrid_hash_reindex(
                path,
                vector_db,
                strategy,
                embedding_adapter
            )


if __name__ == "__main__":
    main()
