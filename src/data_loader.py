import os
import json
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EMBEDDINGS_DIR = BASE_DIR / "embeddings" / "chroma_index"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

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
        
def prepare_chunks(data, chunk_size: int=500, chunk_overlap: int=100):
    """
    Merge question and answer, then split into chunks.
    """
    docs = []
    for item in data:
        combined_text = f"Question: {item["question"]}\nAnswer: {item['answer']}"
        docs.append(combined_text)
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                              chunk_overlap=chunk_overlap)
    chunks = splitter.split_text("\n\n".join(docs))
    
    return chunks

def create_vectorstore(chunks, persist_dir=EMBEDDINGS_DIR):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_dir)
    
    return db
    
def main():
    medquad_path = os.path.join(RAW_DIR, "med_quad.csv")
    hcm_path = os.path.join(RAW_DIR, "HealthCareMagic-100k.json")

    med_data = load_medquad(medquad_path)
    hcm_data = load_healthcaremagic(hcm_path)
    all_data = med_data + hcm_data
    logging.info(f"Total QA pairs loaded: {len(all_data)}")

    with open(os.path.join(PROCESSED_DIR, "qa_combined.json"), "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    chunks = prepare_chunks(all_data)
    logging.info(f"Total text chunks created: {len(chunks)}")

    create_vectorstore(chunks)

if __name__ == "__main__":
    main()
