from enum import Enum
from typing import List, Callable
from pathlib import Path
import numpy as np

from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from models.chunk import Chunk, ChunkMetadata
from config import (
    TOKENIZER_MODEL,
    MAX_TOKENS,
    SLIDING_WINDOW_TOKENS,
    SLIDING_OVERLAP,
    SENTENCE_GROUP,
    SEMANTIC_SIM_THRESHOLD,
)


# =========================
# Strategy Enum
# =========================
class ChunkingStrategy(str, Enum):
    STRUCTURE = "structure"
    SENTENCE = "sentence"
    SLIDING = "sliding"
    SEMANTIC = "semantic"
    HYBRID_SECTION_SEMANTIC = "hybrid_section_semantic"
    AUTO = "auto"


# =========================
# Global tokenizer & chunker
# =========================
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_MODEL),
    max_tokens=MAX_TOKENS,
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,
)


# =========================
# Utilities
# =========================
def normalize_source_id(pdf_path: str) -> str:
    return Path(pdf_path).stem.lower().replace(" ", "_")


def sentence_split(text: str) -> List[str]:
    return [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# =========================
# Chunkers
# =========================
def structure_chunk(doc, source_id: str) -> List[Chunk]:
    chunks = []
    idx = 0

    for c in chunker.chunk(doc):
        text = chunker.contextualize(c).strip()
        if not text:
            continue

        meta = ChunkMetadata(
            section_path=", ".join(map(str, getattr(c.meta, "headings", [])))
            if getattr(c.meta, "headings", None) else None,

            page_numbers=", ".join(
                map(str, sorted(set(getattr(c.meta, "page_numbers", [])))))
            if getattr(c.meta, "page_numbers", None) else None,

            chunk_index=idx,
        )

        chunks.append(
            Chunk(
                source_id=source_id,
                chunk_id=f"{source_id}_{idx}",
                text=text,
                metadata=meta,
            )
        )
        idx += 1

    return chunks


def sentence_chunk(text: str, source_id: str) -> List[Chunk]:
    sentences = sentence_split(text)
    chunks, idx = [], 0
    buf = []

    for s in sentences:
        buf.append(s)
        if len(buf) >= SENTENCE_GROUP:
            chunks.append(
                Chunk(
                    source_id=source_id,
                    chunk_id=f"{source_id}_{idx}",
                    text=". ".join(buf).strip() + ".",
                    metadata=ChunkMetadata(chunk_index=idx),
                )
            )
            idx += 1
            buf = []

    if buf:
        chunks.append(
            Chunk(
                source_id=source_id,
                chunk_id=f"{source_id}_{idx}",
                text=". ".join(buf).strip() + ".",
                metadata=ChunkMetadata(chunk_index=idx),
            )
        )

    return chunks


def sliding_window_chunk(text: str, source_id: str) -> List[Chunk]:
    tokens = tokenizer.tokenizer.tokenize(text)
    chunks, idx = [], 0
    i = 0

    while i < len(tokens):
        window = tokens[i:i + SLIDING_WINDOW_TOKENS]
        chunk_text = tokenizer.tokenizer.convert_tokens_to_string(window)

        chunks.append(
            Chunk(
                source_id=source_id,
                chunk_id=f"{source_id}_{idx}",
                text=chunk_text,
                metadata=ChunkMetadata(chunk_index=idx),
            )
        )
        idx += 1

        if i + SLIDING_WINDOW_TOKENS >= len(tokens):
            break
        i += SLIDING_WINDOW_TOKENS - SLIDING_OVERLAP

    return chunks


# =========================
# Semantic chunk
# =========================
def semantic_chunk(
    text: str,
    source_id: str,
    *,
    embed_fn: Callable[[List[str]], np.ndarray],
    atomic_tokenizer,
    atomic_size: int = 120,
) -> List[Chunk]:

    tokens = atomic_tokenizer.tokenize(text)
    atomic_texts = [
        atomic_tokenizer.convert_tokens_to_string(
            tokens[i:i + atomic_size]
        ).strip()
        for i in range(0, len(tokens), atomic_size)
    ]

    if not atomic_texts:
        return []

    embeddings = np.asarray(embed_fn(atomic_texts))

    chunks, idx = [], 0
    cur_text = atomic_texts[0]
    cur_emb = embeddings[0]

    for i in range(1, len(atomic_texts)):
        sim = cosine_sim(cur_emb, embeddings[i])
        if sim >= SEMANTIC_SIM_THRESHOLD:
            cur_text += " " + atomic_texts[i]
            cur_emb = (cur_emb + embeddings[i]) / 2
        else:
            chunks.append(
                Chunk(
                    source_id=source_id,
                    chunk_id=f"{source_id}_{idx}",
                    text=cur_text.strip(),
                    metadata=ChunkMetadata(chunk_index=idx),
                )
            )
            idx += 1
            cur_text = atomic_texts[i]
            cur_emb = embeddings[i]

    chunks.append(
        Chunk(
            source_id=source_id,
            chunk_id=f"{source_id}_{idx}",
            text=cur_text.strip(),
            metadata=ChunkMetadata(chunk_index=idx),
        )
    )

    return chunks


def hybrid_section_semantic_chunk(
    doc,
    source_id: str,
    *,
    embed_fn,
) -> List[Chunk]:

    final_chunks = []
    global_idx = 0

    for c in chunker.chunk(doc):
        base_text = chunker.contextualize(c).strip()
        if not base_text:
            continue

        token_count = len(tokenizer.tokenizer.tokenize(base_text))

        base_meta = ChunkMetadata(
            section_path=", ".join(map(str, getattr(c.meta, "headings", [])))
            if getattr(c.meta, "headings", None) else None,

            page_numbers=", ".join(
                map(str, sorted(set(getattr(c.meta, "page_numbers", [])))))
            if getattr(c.meta, "page_numbers", None) else None,

            chunk_index=global_idx,
        )

        if token_count <= MAX_TOKENS:
            base_meta.chunk_index = global_idx
            final_chunks.append(
                Chunk(
                    source_id=source_id,
                    chunk_id=f"{source_id}_{global_idx}",
                    text=base_text,
                    metadata=base_meta,
                )
            )
            global_idx += 1
            continue

        sub_chunks = semantic_chunk(
            base_text,
            source_id,
            embed_fn=embed_fn,
            atomic_tokenizer=tokenizer.tokenizer,
        )

        for sub in sub_chunks:
            sub.metadata.section_path = base_meta.section_path
            sub.metadata.page_numbers = base_meta.page_numbers
            sub.metadata.chunk_index = global_idx
            sub.chunk_id = f"{source_id}_{global_idx}"
            final_chunks.append(sub)
            global_idx += 1

    return final_chunks


# =========================
# Dispatcher
# =========================
def extract_full_text_from_doc(doc) -> str:
    texts = []
    for c in chunker.chunk(doc):
        t = chunker.contextualize(c).strip()
        if t:
            texts.append(t)
    return "\n".join(texts)


def chunk_document(
    doc,
    pdf_path: str,
    strategy: ChunkingStrategy,
    *,
    raw_text: str = None,
    embed_fn=None,
) -> List[Chunk]:

    source_id = normalize_source_id(pdf_path)

    if strategy == ChunkingStrategy.STRUCTURE:
        return structure_chunk(doc, source_id)

    if strategy == ChunkingStrategy.HYBRID_SECTION_SEMANTIC:
        if embed_fn is None:
            raise ValueError("HYBRID_SECTION_SEMANTIC requires embed_fn")
        return hybrid_section_semantic_chunk(doc, source_id, embed_fn=embed_fn)

    if raw_text is None:
        raise ValueError(f"{strategy.value} requires raw_text")

    if strategy == ChunkingStrategy.SENTENCE:
        return sentence_chunk(raw_text, source_id)

    if strategy == ChunkingStrategy.SLIDING:
        return sliding_window_chunk(raw_text, source_id)

    if strategy == ChunkingStrategy.SEMANTIC:
        if embed_fn is None:
            raise ValueError("SEMANTIC requires embed_fn")
        return semantic_chunk(
            raw_text,
            source_id,
            embed_fn=embed_fn,
            atomic_tokenizer=tokenizer.tokenizer,
        )

    chunks = structure_chunk(doc, source_id)
    if any(len(tokenizer.tokenizer.tokenize(c.text)) > MAX_TOKENS for c in chunks):
        return hybrid_section_semantic_chunk(doc, source_id, embed_fn=embed_fn)

    return chunks
