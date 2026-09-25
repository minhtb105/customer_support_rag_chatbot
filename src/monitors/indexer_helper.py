"""Helper to reindex a single promoted PDF without full corpus scan"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

def reindex_single_pdf(pdf_path: str, strategies: Optional[list] = None, version_label: str | None = None, publication_date: str | None = None):
    """Reindex single PDF using existing hybrid_hash_reindex logic."""
    try:
        from src.shared.indexer import hybrid_hash_reindex, get_or_create_vectorstore, _resolve_index_db_dir, compute_file_fingerprint
        from src.shared.embedding.adapter import EmbeddingAdapter
        from src.shared.models.chunk import Chunk  # noqa
        from src.shared.chunk_strategies import ChunkingStrategy
        from src.shared.config import META_DB_PATH, EMBEDDING_PROVIDER, EMBEDDING_MODEL, TOKENIZER_MODEL, OPENAI_EMBEDDING_MODEL, BASE_DIR
        from src.shared.metadata_store import init_db
    except ImportError:
        from shared.indexer import hybrid_hash_reindex, get_or_create_vectorstore, _resolve_index_db_dir, compute_file_fingerprint  # type: ignore
        from shared.embedding.adapter import EmbeddingAdapter  # type: ignore
        from shared.models.chunk import Chunk  # type: ignore
        from shared.chunk_strategies import ChunkingStrategy  # type: ignore
        from shared.config import META_DB_PATH, EMBEDDING_PROVIDER, EMBEDDING_MODEL, TOKENIZER_MODEL, OPENAI_EMBEDDING_MODEL, BASE_DIR  # type: ignore
        from shared.metadata_store import init_db  # type: ignore

    init_db(META_DB_PATH)

    if strategies is None:
        strategies = [
            ChunkingStrategy.STRUCTURE,
            ChunkingStrategy.SLIDING,
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.HYBRID_SECTION_SEMANTIC,
        ]

    # Prepare embedding adapter (reuse indexer logic)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from transformers import AutoTokenizer
        try:
            from langchain_openai import OpenAIEmbeddings
            _HAS_OAI = True
        except ImportError:
            _HAS_OAI = False
        import os
        if EMBEDDING_PROVIDER == "openai" and _HAS_OAI:
            try:
                from langchain_openai import OpenAIEmbeddings as _OE
                embedder = _OE(model=OPENAI_EMBEDDING_MODEL)
                try:
                    import tiktoken
                    enc = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL.replace("text-embedding-", "gpt-4"))
                    class _TK:
                        def __call__(self, text, truncation=True, max_length=512, return_tensors=None):
                            toks = enc.encode(text)[:max_length]
                            return {"input_ids": toks}
                        def decode(self, ids, skip_special_tokens=True):
                            return enc.decode(ids)
                    tokenizer = _TK()
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
            except Exception as e:
                print(f"[indexer_helper] OpenAI fallback HF: {e}")
                embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        else:
            embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        from src.shared.embedding.adapter import EmbeddingAdapter as EA  # type: ignore
        embedding_adapter = EA(embedder=embedder, tokenizer=tokenizer, max_len=512) if 'EA' in locals() else None
        # fallback if EA not imported
        if embedding_adapter is None:
            try:
                from shared.embedding.adapter import EmbeddingAdapter as EA2  # type: ignore
                embedding_adapter = EA2(embedder=embedder, tokenizer=tokenizer, max_len=512)
            except ImportError:
                from src.shared.embedding.adapter import EmbeddingAdapter as EA3  # type: ignore
                embedding_adapter = EA3(embedder=embedder, tokenizer=tokenizer, max_len=512)
    except Exception as e:
        raise RuntimeError(f"Embedding adapter init failed: {e}")

    for strategy in strategies:
        db_dir = _resolve_index_db_dir(strategy.value)
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        vector_db = get_or_create_vectorstore(db_dir)
        hybrid_hash_reindex(pdf_path, vector_db, strategy, embedding_adapter, version_label=version_label, publication_date=publication_date)
