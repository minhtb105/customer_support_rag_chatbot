import pandas as pd
from tqdm import tqdm
from typing import List, Dict
import hashlib, os, json, logging
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from langchain_chroma.vectorstores import Chroma
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from docling.document_converter import DocumentConverter
from config import *


# ---------- Directory setup ----------
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PDF_DB_DIR, exist_ok=True)
os.makedirs(QA_DB_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ---------- PDF Parsing + Chunking ----------
def parse_and_chunk_pdf(pdf_path: str) -> List[Dict]:
    """
    Parse a single pdf file -> chunk it using HybridChunker → return a list of chunk dictionaries.
    """
    converter = DocumentConverter()
    doc = converter.convert(pdf_path).document
    
    chunks = []
    for idx, ch in enumerate(chunker.chunk(doc)):
        chunk_dict = {
            "source_id": os.path.basename(pdf_path),
            "chunk_id": f"{os.path.basename(pdf_path)}_chunk_{idx+1}",
            "text": ch.text.strip(),
            "metadata": {
                "section_path": ch.metadata.get("section_path", []),
                "page_numbers": ch.metadata.get("page_numbers", []),
                "chunk_index": idx + 1,
            },
        }
        chunks.append(chunk_dict)

    return chunks

def process_pdf_folder(pdf_dir: str = PDF_DIR, output_dir: str = OUTPUT_DIR):
    """
    Iterate through all PDFs in a folder → parse + chunk each → save results as JSON files.
    """
    pdf_files = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]

    logging.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            chunks = parse_and_chunk_pdf(pdf_path)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_chunks.json")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            logging.info(f"Saved {len(chunks)} chunks → {out_path}")

        except Exception as e:
            logging.info(f"Error processing {pdf_path}: {e}")

# ---------- Dataset Loading ----------
def load_medquad(file_path: str):
    df = pd.read_csv(file_path)
    data = []
    
    for _, row in tqdm(df.iterrows()):
        q = str(row.get("Question", "")).strip()
        a = str(row.get("Answer", "")).strip()
        
        if q and a:
            data.append({"source": "MedQuAD", "question": q, "answer": a})

    return data

def load_healthcaremagic(file_path: str):
    with open(file_path, 'r', encoding="utf-8") as f:
        healthcare_data = json.load(f)
        
    data = []
    for item in tqdm(healthcare_data):
        q = str(item.get("input", "")).strip()
        a = str(item.get("output", "")).strip()
        
        if q and a:
            data.append({"source": "HealthCareMagic", "question": q, "answer": a})
            
    return data
        
# ---------- Text Chunk Preparation ----------
def prepare_chunks(data):
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

# ---------- Vectorstore Creation ----------
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    
def needs_reindex(path: str, metadata_path: str="processed/embeddings_metadata.json"):
    if not os.path.exists(metadata_path): return True
    
    with open(metadata_path, "r") as f:
        meta = json.load(f)
        
    old_hash = meta.get(hash)
    new_hash = file_hash(path)
    
    return old_hash != new_hash
    
def create_vectorstore(texts, metadatas, db_dir):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=db_dir,
        metadatas=metadatas
    )
    db.persist()
    
    return db
    
# ---------- Main Pipeline ----------
def main():
    # Step 1: Process PDFs into hierarchical chunks
    process_pdf_folder(PDF_DIR, OUTPUT_DIR)
    
    # Step 2: Load existing QA datasets
    medquad_path = os.path.join(RAW_DIR, "med_quad.csv")
    hcm_path = os.path.join(RAW_DIR, "HealthCareMagic-100k.json")

    med_data = load_medquad(medquad_path)
    hcm_data = load_healthcaremagic(hcm_path)
    all_data = med_data + hcm_data
    logging.info(f"Total QA pairs loaded: {len(all_data)}")

    # Step 3: Split into token chunks
    with open(os.path.join(PROCESSED_DIR, "qa_combined.json"), "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    texts, metadatas = prepare_chunks(all_data)
    logging.info(f"Total text chunks created: {len(texts)}")

    # Step 4: Build vectorstore
    qa_db = create_vectorstore(texts, metadatas, QA_DB_DIR)
    
    pdf_json_files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith("_chunks.json")]
    pdf_texts, pdf_metas = [], []
    for fpath in pdf_json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                pdf_texts.append(item["text"])
                pdf_metas.append({
                    "source_id": item["source_id"],
                    "section_path": item["metadata"]["section_path"],
                    "page_numbers": item["metadata"]["page_numbers"],
                    "source_type": "pdf"
                })
    pdf_db = create_vectorstore(pdf_texts, pdf_metas, PDF_DB_DIR)

if __name__ == "__main__":
    main()
