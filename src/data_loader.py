import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
import os, hashlib, logging, json
from langchain_chroma.vectorstores import Chroma
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from docling.document_converter import DocumentConverter
from metadata_store import (
    init_db,
    upsert_file_and_chunks,
    get_chunk_hashes_for_file,
    delete_chunks_by_hashes,
    get_file_hash,
    find_vector_ids_for_chunk_hashes
)
from config import *


# ---------- Directory setup ----------
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PDF_DB_DIR, exist_ok=True)
os.makedirs(QA_DB_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Tokenizer & Chunkers ----------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

splitter = TokenTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=CHUNK_SIZE,
    merge_peers=MERGE_PEERS
)

init_db(META_DB_PATH)

# ---------- PDF Parsing + Chunking ----------
def parse_and_chunk_pdf(pdf_path: str) -> List[Dict]:
    """
    Parse a single pdf file → chunk it using HybridChunker → return a list of chunk dictionaries.
    """
    converter = DocumentConverter()
    doc = converter.convert(pdf_path).document

    chunks = []
    for idx, ch in enumerate(chunker.chunk(doc)):
        meta = ch.meta
        section_path = getattr(meta, "headings", [])

        # Extract page numbers
        page_numbers = []
        if hasattr(meta, "doc_items"):
            for item in meta.doc_items:
                if hasattr(item, "prov"):
                    for prov_item in item.prov:
                        page_numbers.append(prov_item.page_no)

        chunk_dict = {
            "source_id": os.path.basename(pdf_path),
            "chunk_id": f"{os.path.basename(pdf_path)}_chunk_{idx+1}",
            "text": ch.text.strip(),
            "metadata": {
                "section_path": section_path,
                "page_numbers": sorted(set(page_numbers)),
                "chunk_index": idx + 1,
            },
        }
        chunks.append(chunk_dict)

    return chunks

def clean_metadata(meta: dict):
    clean_meta = {}
    for k, v in meta.items():
        if isinstance(v, list):
            # join list thành chuỗi
            clean_meta[k] = ", ".join(map(str, v))
        elif isinstance(v, dict):
            # chuyển dict thành JSON string
            clean_meta[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean_meta[k] = v
            
    return clean_meta

# ---------- Dataset Loading ----------
def load_medquad(file_path: str):
    df = pd.read_csv(file_path)
    data = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q = str(row.get("Question", "")).strip()
        a = str(row.get("Answer", "")).strip()
        if q and a:
            data.append({"source": "MedQuAD", "question": q, "answer": a})

    return data


def load_healthcaremagic(file_path: str):
    df = pd.read_json(file_path)
    data = []
    for item in df.iterrows():
        q = str(item[1]['input']).strip()
        a = str(item[1]['output']).strip()
        if q and a:
            data.append({"source": "HealthCareMagic", "question": q, "answer": a})

    return data


# ---------- Text Chunk Preparation ----------
def prepare_chunks(data: List[Dict]):
    texts, metadatas = [], []
    for i, item in enumerate(data):
        q = item["question"].strip()
        a = item["answer"].strip()
        combined_text = f"Question: {q}\nAnswer: {a}"
        parts = splitter.split_text(combined_text)
        for part in parts:
            texts.append(part)
            metadatas.append({"source_id": f"{item['source']}_{i}"})
            
    return texts, metadatas


# ---------- Vectorstore Utilities ----------
def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def compute_chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def hybrid_hash_reindex(path: str, chunk_func, vector_db):
    """
    Reindex a file with metadata stored in SQLite.
    - path: path to the file (used to compute file hash and as key)
    - chunk_func: function(path) -> either:
        * list[dict] where each dict has "text" and optional "metadata", OR
        * (texts: List[str], metadatas: List[Dict])
    - vector_db: vector database (e.g., Chroma) that supports .add_texts(...) and .delete(ids=[...])
    """
    # read file and compute file hash
    try:
        with open(path, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        logging.error(f"Cannot read file {path}: {e}")
        return

    file_hash = compute_file_hash(file_bytes)
    fname = os.path.basename(path)

    # get stored file hash (None if not present)
    stored_file_hash = get_file_hash(fname)
    if stored_file_hash == file_hash:
        # file unchanged — skip (fast path)
        logging.info(f"No file-level changes detected for {fname} (file_hash unchanged). Skipping reindex.")
        return

    # file is new or changed -> produce chunks and their hashes
    result = chunk_func(path)
    # normalize result to (texts, metadatas)
    texts, metadatas = [], []
    if (isinstance(result, tuple) or isinstance(result, list)) and len(result) == 2 and isinstance(result[0], list):
        texts, metadatas = result
    elif isinstance(result, list):
        for ch in result:
            if isinstance(ch, dict) and "text" in ch:
                texts.append(ch["text"])
                meta = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
                metadatas.append(meta)
            else:
                texts.append(str(ch))
                metadatas.append({})
    else:
        logging.error("chunk_func returned unsupported type; expected list or (texts, metadatas).")
        return

    new_hashes = [compute_chunk_hash(t) for t in texts]

    # get old chunk hashes (may be empty list if file is new)
    old_hashes = get_chunk_hashes_for_file(fname) or []

    # determine removed chunks (present before, absent now)
    removed_hashes = list(set(old_hashes) - set(new_hashes))

    # determine which new chunks to add (and changed chunks)
    # we treat any new_hash not in old_hashes as "to add or update"
    added_or_changed_indices = [i for i, h in enumerate(new_hashes) if h not in old_hashes]

    # --- delete removed chunks from vectorstore + metadata DB ---
    if removed_hashes:
        logging.info(f"{fname}: {len(removed_hashes)} chunk(s) removed -> deleting from vector DB and metadata DB.")
        del_ids = find_vector_ids_for_chunk_hashes(removed_hashes)
        if del_ids:
            try:
                vector_db.delete(ids=del_ids)
            except Exception as e:
                logging.error(f"Failed to delete vectors for removed chunks of {fname}: {e}")
        # remove rows from chunks table
        try:
            delete_chunks_by_hashes(removed_hashes)
        except Exception as e:
            logging.error(f"Failed to delete chunk rows for {fname}: {e}")

    # --- add/update new or changed chunks ---
    new_chunks_rows = []
    if added_or_changed_indices:
        logging.info(f"{fname}: adding/updating {len(added_or_changed_indices)} chunk(s) to vector DB.")
        # batch insert for performance
        for batch_start in range(0, len(added_or_changed_indices), BATCH_SIZE):
            batch_indices = added_or_changed_indices[batch_start: batch_start + BATCH_SIZE]
            batch_texts, batch_metas, batch_ids = [], [], []

            for i in batch_indices:
                text = texts[i]
                meta = metadatas[i] if i < len(metadatas) else {}
                meta = clean_metadata(meta)
                vector_id = f"{fname}_chunk_{i}_{new_hashes[i][:12]}"
                batch_texts.append(text)
                batch_metas.append(meta)
                batch_ids.append(vector_id)
                new_chunks_rows.append({
                    "chunk_hash": new_hashes[i],
                    "chunk_index": i,
                    "vector_id": vector_id,
                    "extra_meta": meta
                })

            try:
                # Add all chunks in a batch
                vector_db.add_texts(batch_texts, metadatas=batch_metas, ids=batch_ids)
                logging.info(f"Inserted batch {batch_start // BATCH_SIZE + 1} "
                            f"({len(batch_indices)} chunks) for {fname}")
            except Exception as e:
                logging.error(f"Failed to add batch starting at index {batch_start} for {fname}: {e}")

    # --- Upsert file record + the new/updated chunk rows into metadata DB
    # This will create or update files.file_hash and insert/update chunk rows
    try:
        upsert_file_and_chunks(fname, file_hash, new_chunks_rows)
    except Exception as e:
        logging.error(f"Failed to upsert metadata for {fname}: {e}")

    logging.info(f"Reindex complete for {fname}: removed={len(removed_hashes)}, added_or_changed={len(added_or_changed_indices)}")

def get_or_create_vectorstore(db_dir: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    if os.path.exists(os.path.join(db_dir, "chroma.sqlite3")):
        logging.info(f"Loading existing vectorstore from {db_dir}")
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    else:
        logging.info(f"Creating new (empty) vectorstore at {db_dir}")
        os.makedirs(db_dir, exist_ok=True)
        # initialize empty Chroma collection manually
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
    return db

# ---------- Main Pipeline ----------
def main():
    # Step 1: Load QA datasets
    medquad_path = os.path.join(RAW_DIR, "med_quad.csv")
    hcm_path = os.path.join(PROCESSED_DIR, "HealthCareMagic-100k.json")
    
    med_data = load_medquad(medquad_path)
    hcm_data = load_healthcaremagic(hcm_path)
    
    # Step 2: Reindex QA datasets incrementally
    qa_db = get_or_create_vectorstore(QA_DB_DIR)
    hybrid_hash_reindex(medquad_path, lambda _: prepare_chunks(med_data), qa_db)
    hybrid_hash_reindex(hcm_path, lambda _: prepare_chunks(hcm_data), qa_db)

    # Step 3: PDF reindex
    pdf_files = [
        os.path.join(PDF_DIR, f)
        for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    pdf_db = get_or_create_vectorstore(PDF_DB_DIR)
    for pdf_path in pdf_files:
        hybrid_hash_reindex(pdf_path, parse_and_chunk_pdf, pdf_db)

if __name__ == "__main__":
    main()
    