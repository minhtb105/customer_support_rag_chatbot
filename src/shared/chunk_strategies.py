from enum import Enum as _Enum  # fallback only; canonical ChunkingStrategy lives in models.enums
from typing import List, Callable, Dict, Tuple
from functools import lru_cache
from pathlib import Path
import numpy as np

from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

try:
    from shared.models.enums import ChunkingStrategy
except ImportError:
    try:
        from src.shared.models.enums import ChunkingStrategy  # type: ignore
    except ImportError:
        class ChunkingStrategy(str, _Enum):  # fallback when models package unavailable
            STRUCTURE = "structure"
            SENTENCE = "sentence"
            SLIDING = "sliding"
            SEMANTIC = "semantic"
            HYBRID_SECTION_SEMANTIC = "hybrid_section_semantic"
            AUTO = "auto"

try:
    from shared.models.chunk import Chunk, ChunkMetadata
    from shared.config import (
        TOKENIZER_MODEL,
        MAX_TOKENS,
        SLIDING_WINDOW_TOKENS,
        SLIDING_OVERLAP,
        SENTENCE_GROUP,
        SEMANTIC_SIM_THRESHOLD,
        ATOMIC_TOKEN_SIZE
    )
except ImportError:
    from src.shared.models.chunk import Chunk, ChunkMetadata
    from src.shared.config import (
        TOKENIZER_MODEL,
        MAX_TOKENS,
        SLIDING_WINDOW_TOKENS,
        SLIDING_OVERLAP,
        SENTENCE_GROUP,
        SEMANTIC_SIM_THRESHOLD,
        ATOMIC_TOKEN_SIZE
    )


# =========================
# Strategy Enum (canonical in shared.models.enums, re-exported here)
# =========================
try:
    ChunkingStrategy  # noqa: B018 — provided by the import above
except NameError:  # pragma: no cover — fallback defined above already
    pass


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


@lru_cache(maxsize=8)
def _get_chunker(max_tokens: int) -> HybridChunker:
    """Chunker phụ cho structure-variants (MAX_TOKENS khác default)."""
    tok = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_MODEL),
        max_tokens=max_tokens,
    )
    return HybridChunker(tokenizer=tok, merge_peers=True)


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

def serialize_metatdata(meta: dict):
    out = {}
    for k, v in meta.items():
        if isinstance(v, list):
            out[k] = "|||".join(map(str, v))
        else:
            out[k] = v

    return out


# Correct-spelling alias (old typo kept importable for indexer/nutrition callers).
serialize_metadata = serialize_metatdata


# =========================
# Chunkers
# =========================
def structure_chunk(doc, source_id: str, *, max_tokens: int = None) -> List[Chunk]:
    chunks = []
    idx = 0
    ch = _get_chunker(max_tokens) if max_tokens else chunker

    for c in ch.chunk(doc):
        text = chunker.contextualize(c).strip()
        if not text:
            continue
        
        headings = getattr(c.meta, "headings", None)
        pages = getattr(c.meta, "page_numbers", None)

        meta = ChunkMetadata(
            section_path=list(headings) if headings else None,
            page_numbers=list(pages) if pages else None,
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


def sentence_chunk(text: str, source_id: str, *, group_size: int = None) -> List[Chunk]:
    sentences = sentence_split(text)
    chunks, idx = [], 0
    buf = []
    group = group_size if group_size else SENTENCE_GROUP

    for s in sentences:
        buf.append(s)
        if len(buf) >= group:
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


def sliding_window_chunk(text: str, source_id: str, *, window_tokens: int = None, overlap: int = None) -> List[Chunk]:
    tokens = tokenizer.tokenizer.tokenize(text)
    chunks, idx = [], 0
    i = 0
    win = window_tokens if window_tokens else SLIDING_WINDOW_TOKENS
    ov = overlap if overlap is not None else SLIDING_OVERLAP

    while i < len(tokens):
        window = tokens[i:i + win]
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

        if i + win >= len(tokens):
            break
        i += win - ov

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
    sim_threshold: float = None,
) -> List[Chunk]:
    thr = sim_threshold if sim_threshold is not None else SEMANTIC_SIM_THRESHOLD

    tokens = atomic_tokenizer.tokenize(text)
    atomic_texts = [
        atomic_tokenizer.convert_tokens_to_string(
            tokens[i:i + atomic_size]
        ).strip()
        for i in range(0, len(tokens), atomic_size)
    ]

    if not atomic_texts:
        return []

    # Bỏ atomic trắng: NVIDIA /v1/embeddings 400 với input rỗng.
    atomic_texts = [t for t in atomic_texts if t.strip()]
    if not atomic_texts:
        return []

    embeddings = np.asarray(embed_fn(atomic_texts))

    chunks, idx = [], 0
    cur_text = atomic_texts[0]
    cur_emb = embeddings[0]

    for i in range(1, len(atomic_texts)):
        sim = cosine_sim(cur_emb, embeddings[i])
        if sim >= thr:
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
    max_tokens: int = None,
    atomic_size: int = None,
    sim_threshold: float = None,
) -> List[Chunk]:

    final_chunks = []
    global_idx = 0
    cap = max_tokens if max_tokens else MAX_TOKENS
    ch = _get_chunker(cap) if max_tokens else chunker

    for c in ch.chunk(doc):
        base_text = ch.contextualize(c).strip()
        if not base_text:
            continue

        token_count = len(tokenizer.tokenizer.tokenize(base_text))

        headings = getattr(c.meta, "headings", None)
        pages = getattr(c.meta, "page_numbers", None)

        base_meta = ChunkMetadata(
            section_path=list(headings) if headings else None,
            page_numbers=list(pages) if pages else None,
            chunk_index=global_idx,
        )

        if token_count <= cap:
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
            atomic_size=atomic_size if atomic_size else ATOMIC_TOKEN_SIZE,
            sim_threshold=sim_threshold,
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
    **overrides,
) -> List[Chunk]:

    source_id = normalize_source_id(pdf_path)

    if strategy == ChunkingStrategy.STRUCTURE:
        if overrides and set(overrides) != {"max_tokens"}:
            raise ValueError(f"STRUCTURE accepts only max_tokens, got {sorted(overrides)}")
        return structure_chunk(doc, source_id, **overrides)

    if strategy == ChunkingStrategy.HYBRID_SECTION_SEMANTIC:
        if embed_fn is None:
            raise ValueError("HYBRID_SECTION_SEMANTIC requires embed_fn")
        if overrides and not set(overrides) <= {"max_tokens", "atomic_size", "sim_threshold"}:
            raise ValueError(f"HYBRID accepts max_tokens/atomic_size/sim_threshold, got {sorted(overrides)}")
        return hybrid_section_semantic_chunk(doc, source_id, embed_fn=embed_fn, **overrides)

    if raw_text is None:
        raise ValueError(f"{strategy.value} requires raw_text")

    if strategy == ChunkingStrategy.SENTENCE:
        if overrides and set(overrides) != {"group_size"}:
            raise ValueError(f"SENTENCE accepts only group_size, got {sorted(overrides)}")
        return sentence_chunk(raw_text, source_id, **overrides)

    if strategy == ChunkingStrategy.SLIDING:
        if overrides and not set(overrides) <= {"window_tokens", "overlap"}:
            raise ValueError(f"SLIDING accepts window_tokens/overlap, got {sorted(overrides)}")
        return sliding_window_chunk(raw_text, source_id, **overrides)

    if strategy == ChunkingStrategy.SEMANTIC:
        if embed_fn is None:
            raise ValueError("SEMANTIC requires embed_fn")
        if overrides and not set(overrides) <= {"atomic_size", "sim_threshold"}:
            raise ValueError(f"SEMANTIC accepts atomic_size/sim_threshold, got {sorted(overrides)}")
        return semantic_chunk(
            raw_text,
            source_id,
            embed_fn=embed_fn,
            atomic_tokenizer=tokenizer.tokenizer,
            atomic_size=overrides.get("atomic_size", ATOMIC_TOKEN_SIZE),
            sim_threshold=overrides.get("sim_threshold"),
        )

    chunks = structure_chunk(doc, source_id)
    if any(len(tokenizer.tokenizer.tokenize(c.text)) > MAX_TOKENS for c in chunks):
        return hybrid_section_semantic_chunk(doc, source_id, embed_fn=embed_fn)

    return chunks


# =========================
# Variant registry (D1)
# =========================
# variant name -> (base enum, param overrides). Index dir = registry name.
VARIANTS: Dict[str, Tuple[ChunkingStrategy, dict]] = {
    "structure": (ChunkingStrategy.STRUCTURE, {}),
    "sentence": (ChunkingStrategy.SENTENCE, {}),
    "sliding": (ChunkingStrategy.SLIDING, {}),
    "semantic": (ChunkingStrategy.SEMANTIC, {}),
    "hybrid_section_semantic": (ChunkingStrategy.HYBRID_SECTION_SEMANTIC, {}),
    # Stage-2 variants của top-2 base (B1: hybrid 0.8875, sentence 0.8625).
    "hybrid__t256": (ChunkingStrategy.HYBRID_SECTION_SEMANTIC, {"max_tokens": 256}),
    "hybrid__t768": (ChunkingStrategy.HYBRID_SECTION_SEMANTIC, {"max_tokens": 768}),
    "hybrid__thr080": (ChunkingStrategy.HYBRID_SECTION_SEMANTIC, {"sim_threshold": 0.80}),
    "hybrid__thr095": (ChunkingStrategy.HYBRID_SECTION_SEMANTIC, {"sim_threshold": 0.95}),
    "sentence__g2": (ChunkingStrategy.SENTENCE, {"group_size": 2}),
    "sentence__g6": (ChunkingStrategy.SENTENCE, {"group_size": 6}),
    "sentence__g8": (ChunkingStrategy.SENTENCE, {"group_size": 8}),
}


def resolve_strategy(name: str) -> Tuple[ChunkingStrategy, dict]:
    """Resolve registry name -> (base, overrides). Fail LOUD on unknown."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown chunking strategy: {name!r} (known: {sorted(VARIANTS)})"
        )
