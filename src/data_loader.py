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
from models.chunk import Chunk, ChunkMetadata


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

        chunks.append(Chunk(
            source_id=os.path.basename(pdf_path),
            chunk_id=f"{os.path.basename(pdf_path)}_chunk_{idx+1}",
            text=ch.text.strip(),
            metadata=ChunkMetadata(
                section_path=", ".join(map(str, section_path)) if section_path else None,
                page_numbers=", ".join(map(str, sorted(set(page_numbers)))),
                chunk_index=idx + 1
            )
        ))

    return chunks

def clean_metadata(meta: dict):
    clean_meta = {}
    for k, v in meta.items():
        if isinstance(v, list):            
            clean_meta[k] = ", ".join(map(str, v))
        elif isinstance(v, dict):
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
    """
    Convert QA dataset (e.g., MedQuAD, HealthCareMagic)
    into standardized Chunk objects compatible with vectorstore.
    """
    chunks = []
    for i, item in enumerate(data):
        q = item["question"].strip()
        a = item["answer"].strip()
        combined_text = f"Question: {q}\nAnswer: {a}"
        parts = splitter.split_text(combined_text)
        
        for j, part in enumerate(parts):
            chunk = Chunk(
                source_id=f"{item['source']}_{i}",
                chunk_id=f"{item['source']}_{i}_chunk_{j+1}",
                text=part,
                metadata=ChunkMetadata(
                    chunk_index=j + 1
                )
            )
            chunks.append(chunk)
            
    return chunks


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
        logging.info(f"No file-level changes detected for {fname} (file_hash unchanged). Skipping reindex.")
        return

    # --- Step 1: Generate new chunks ---
    try:
        chunks = chunk_func(path)  # expect List[Chunk]
    except Exception as e:
        logging.error(f"Failed to chunk {fname}: {e}")
        return

    if not chunks:
        logging.warning(f"No chunks generated for {fname}. Skipping.")
        return
    
    # --- Step 2: Compute chunk hashes ---
    new_hashes = [compute_chunk_hash(c.text) for c in chunks]

    # get old chunk hashes (may be empty list if file is new)
    old_hashes = get_chunk_hashes_for_file(fname) or []

    # determine removed chunks (present before, absent now)
    removed_hashes = list(set(old_hashes) - set(new_hashes))

    # determine which new chunks to add (and changed chunks)
    # we treat any new_hash not in old_hashes as "to add or update"
    added_or_changed_indices = [i for i, h in enumerate(new_hashes) if h not in old_hashes]

    # --- Step 3: Handle removals ---
    if removed_hashes:
        logging.info(f"{fname}: {len(removed_hashes)} chunk(s) removed -> deleting from vector DB and metadata DB.")
        try:
            del_ids = find_vector_ids_for_chunk_hashes(removed_hashes)
            if del_ids:
                vector_db.delete(ids=del_ids)
            delete_chunks_by_hashes(removed_hashes)
        except Exception as e:
            logging.error(f"Failed to delete outdated chunks for {fname}: {e}")

    # --- Step 4: Add or update chunks ---
    new_chunks_rows = []
    if added_or_changed_indices:
        logging.info(f"{fname}: adding/updating {len(added_or_changed_indices)} chunk(s) to vector DB.")
        for i in added_or_changed_indices:
            c = chunks[i]
            vector_id = f"{fname}_{c.chunk_id}_{new_hashes[i][:12]}"
            meta_dict = c.metadata.model_dump() | {"source_id": c.source_id}
            new_chunks_rows.append({
                "chunk_hash": new_hashes[i],
                "chunk_index": c.metadata.chunk_index,
                "vector_id": vector_id,
                "extra_meta": meta_dict
            })
        
        try:
            # batch insert into vectorstore
            vector_db.add_texts(
                [c.text for c in [chunks[i] for i in added_or_changed_indices]],
                metadatas=[c.metadata.model_dump() | {"source_id": c.source_id} for c in [chunks[i] for i in added_or_changed_indices]],
                ids=[f"{fname}_{c.chunk_id}_{new_hashes[i][:12]}" for i, c in enumerate([chunks[i] for i in added_or_changed_indices])]
            )
        except Exception as e:
            logging.error(f"Failed to insert chunks for {fname}: {e}")
            
    # --- Step 5: Update metadata DB ---
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
    # main()
    parse_and_chunk_pdf(r"data\raw\pdfs\9241594217_eng.pdf")
